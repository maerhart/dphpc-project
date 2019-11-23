
#include <cuda.h>
#include <cooperative_groups.h>

#include <iostream>
#include <vector>
#include <cassert>
#include <fstream>

#include <unistd.h>

using namespace cooperative_groups;
using namespace std;

#define MANAGED_OWNER_NONE -1
#define MANAGED_OWNER_HOST -2

#define CUDA_CHECK(expr) do {\
    cudaError_t err = (expr);\
    if (err != cudaSuccess) {\
        std::cerr << "CUDA ERROR: " << __FILE__ << ":" << __LINE__ << ": " << #expr << " <" << cudaGetErrorName(err) << "> " << cudaGetErrorString(err) << "\n"; \
        abort(); \
    }\
} while(0)

__host__ __device__ void memcpy_volatile(void *dst, void *src, size_t n)
{
    volatile char *d = (char*) dst;
    volatile char *s = (char*) src;
    for (size_t i = 0; i < n; i++) {
        d[i] = s[i];
    }
}

__host__ __device__ void p2pSendHostToDevice(void* ptr, int size, int deviceThread,
    char* managedBuffer, volatile int* managedBufferOwner)
{
#if defined(__CUDA_ARCH__)
    // step 2

    int rank = this_multi_grid().thread_rank();
    if (rank != deviceThread) return;

    //printf("p2pSendHostToDevice device wait start\n");
    while (*managedBufferOwner != deviceThread) {}
    //printf("p2pSendHostToDevice device wait finish\n");

    memcpy_volatile(ptr, managedBuffer, size);
    __threadfence_system();
    *managedBufferOwner = MANAGED_OWNER_NONE;
#else
    // step 1

    //printf("p2pSendHostToDevice host wait start\n");
    while (*managedBufferOwner != MANAGED_OWNER_NONE) {}
    //printf("p2pSendHostToDevice host wait finish\n");

    memcpy_volatile(managedBuffer, ptr, size);
    __sync_synchronize();

    *managedBufferOwner = deviceThread;
#endif
}

__host__ __device__ void p2pSendDeviceToHost(void* ptr, int size, int deviceThread,
    char* managedBuffer, volatile int* managedBufferOwner)
{
#if defined(__CUDA_ARCH__)
    // step 1

    int rank = this_multi_grid().thread_rank();
    if (rank != deviceThread) return;

    //printf("p2pSendDeviceToHost device wait start\n");
    while (*managedBufferOwner != MANAGED_OWNER_NONE) {}
    //printf("p2pSendDeviceToHost device wait finish\n");

    memcpy_volatile(managedBuffer, ptr, size);
    __threadfence_system();
    *managedBufferOwner = MANAGED_OWNER_HOST;
#else
    // step 2

    //printf("p2pSendDeviceToHost host wait start\n");
    while (*managedBufferOwner != MANAGED_OWNER_HOST) {}
    //printf("p2pSendDeviceToHost host wait finish\n");

    memcpy_volatile(ptr, managedBuffer, size);
    __sync_synchronize();

    *managedBufferOwner = MANAGED_OWNER_NONE;

#endif
}

#define PREPARATION 20
#define REPETITIONS 100

__global__ void kernelBenchmarkHostDevice(size_t dataSize, char* deviceSrcData, char* deviceDstData, int peakClkKHz, 
                                          char* managedBuffer, int* managedBufferOwner, double* rawData)
{
    int size = this_grid().size();
    int rank = this_grid().thread_rank(); // or this_multi_grid().grid_rank()

    double E_T = 0;
    double E_T2 = 0;

    double E_BW = 0;
    double E_BW2 = 0;

    int dataPerSendRecv = 2 * dataSize;

    for (int r = 0; r < PREPARATION; r++) {
        auto t1 = clock64();
        p2pSendDeviceToHost(deviceSrcData, dataSize, 0, managedBuffer, managedBufferOwner);
        p2pSendHostToDevice(deviceDstData, dataSize, 0, managedBuffer, managedBufferOwner);
        auto t2 = clock64();
        double dt = (t2 - t1) * 0.001 / peakClkKHz;
        rawData[r] = dt;
    }

    for (int r = 0; r < REPETITIONS; r++) {

        auto t1 = clock64();
        p2pSendDeviceToHost(deviceSrcData, dataSize, 0, managedBuffer, managedBufferOwner);
        p2pSendHostToDevice(deviceDstData, dataSize, 0, managedBuffer, managedBufferOwner);
        auto t2 = clock64();

        double dt = (t2 - t1) * 0.001 / peakClkKHz;
        double bw = dataPerSendRecv / dt;

        rawData[PREPARATION+r] = dt;

        E_T += dt;
        E_T2 += dt * dt;

        E_BW += bw;
        E_BW2 += bw * bw;
    }

    E_T /= REPETITIONS;
    E_T2 /= REPETITIONS;

    E_BW /= REPETITIONS;
    E_BW2 /= REPETITIONS;

    double timePerSendRecv = E_T;
    double timePerSendRecvErr = sqrt(E_T2 - E_T * E_T);
    double bandwidth = E_BW;
    double bandwidthErr = sqrt(E_BW2 - E_BW * E_BW);
    
    printf("dataSize = %d B, time = %lg ( +- %lg ) us, bandwidth = %lg ( +- %lg ) MB/s \n",
               int(dataSize), timePerSendRecv * 1e6, timePerSendRecvErr * 1e6, bandwidth / 1e6, bandwidthErr / 1e6);
}


int main() {
    int deviceCount = -1;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    assert(deviceCount >= 1);
    
    int peakClkKHz = -1;
    CUDA_CHECK(cudaDeviceGetAttribute(&peakClkKHz, cudaDevAttrClockRate, /*device = */0));
    assert(peakClkKHz > 0);
    
    int iterations = 21;

    double* rawData = nullptr;
    CUDA_CHECK(cudaMallocManaged(&rawData, sizeof(double) * (PREPARATION+REPETITIONS)));
    
    // remove file content
    fstream("host_device_raw_data.txt", fstream::out | fstream::trunc);

    for (size_t iteration = 0; iteration < iterations; iteration++) {
        size_t dataSize = 1 << iteration;

        char* deviceSrcData = nullptr;
        CUDA_CHECK(cudaMalloc(&deviceSrcData, dataSize));
        
        char* deviceDstData = nullptr;
        CUDA_CHECK(cudaMalloc(&deviceDstData, dataSize));
        
        char* managedBuffer = nullptr;
        CUDA_CHECK(cudaMallocManaged(&managedBuffer, dataSize));
        //CUDA_CHECK(cudaMemAdvise(managedBuffer, dataSize, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));

        int* managedBufferOwner = nullptr;
        CUDA_CHECK(cudaMallocManaged(&managedBufferOwner, sizeof(int)));
        //CUDA_CHECK(cudaMemAdvise(managedBufferOwner, sizeof(int), cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));


        *managedBufferOwner = MANAGED_OWNER_NONE;
        
        char* hostSrcData = (char*) calloc(dataSize, 1);
        assert(hostSrcData);
        char* hostDstData = (char*) calloc(dataSize, 1);
        assert(hostDstData);
        
        for (int i = 0; i < dataSize; i++) hostSrcData[i] = 17;
        
        CUDA_CHECK(cudaMemcpy(deviceSrcData, hostSrcData, dataSize, cudaMemcpyHostToDevice));
        
        dim3 blocksPerGrid = 1;
        dim3 threadsPerBlock = 1;
        
        void* params[] = {
            (void*)&dataSize,
            (void*)&deviceSrcData,
            (void*)&deviceDstData,
            (void*)&peakClkKHz,
            (void*)&managedBuffer,
            (void*)&managedBufferOwner,
            (void*)&rawData
        };

        void* hostTempBuffer = calloc(dataSize, 1);

        CUDA_CHECK(cudaLaunchCooperativeKernel(
            /*const T *func*/(void*) kernelBenchmarkHostDevice,
            /*dim3 gridDim*/blocksPerGrid,
            /*dim3 blockDim*/threadsPerBlock,
            /*void **args*/params
            /*size_t sharedMem = 0,
            cudaStream_t stream = 0*/
            ));
        
        CUDA_CHECK(cudaPeekAtLastError());

        for (int r = 0; r < (PREPARATION+REPETITIONS); r++) {
            p2pSendDeviceToHost(hostTempBuffer, dataSize, 0, managedBuffer, managedBufferOwner);
            p2pSendHostToDevice(hostTempBuffer, dataSize, 0, managedBuffer, managedBufferOwner);
        }

        CUDA_CHECK(cudaDeviceSynchronize());
        
        CUDA_CHECK(cudaMemcpy(hostDstData, deviceDstData, dataSize, cudaMemcpyDeviceToHost));
        
        for (int i = 0; i < dataSize; i++) {
            if (hostDstData[i] != 17) {
                printf("Incorrect data!");
                abort();
            }
        }

        free(hostTempBuffer);

        free(hostDstData);
        free(hostSrcData);

        {
            fstream f("host_device_raw_data.txt", fstream::out | fstream::app);
            f << dataSize;
            for (int r = 0; r < (PREPARATION+REPETITIONS); r++) {
                f << " " << rawData[r];
            }
            f << endl;
        }

        CUDA_CHECK(cudaFree(managedBuffer));
        CUDA_CHECK(cudaFree(managedBufferOwner));

        CUDA_CHECK(cudaFree(deviceDstData));
        CUDA_CHECK(cudaFree(deviceSrcData));
    }
    
    printf("Exit\n");
    
    return 0;
}
