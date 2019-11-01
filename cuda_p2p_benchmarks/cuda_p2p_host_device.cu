
#include <cuda.h>
#include <cooperative_groups.h>

#include <iostream>
#include <vector>
#include <cassert>

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

__host__ __device__ void memcpy_volatile(volatile void *dst, volatile void *src, size_t n)
{
    volatile char *d = (char*) dst;
    volatile char *s = (char*) src;
    for (size_t i = 0; i < n; i++) {
        d[i] = s[i];
    }
}

__host__ __device__ void p2pSendHostToDevice(void* ptr, int size, int deviceThread,
    volatile char* managedBuffer, volatile int* managedBufferOwner)
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
    volatile char* managedBuffer, volatile int* managedBufferOwner)
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

#define REPETITIONS 10

__global__ void kernelBenchmarkHostDevice(size_t dataSize, char* deviceSrcData, char* deviceDstData, int peakClkKHz, 
                                          volatile char* managedBuffer, volatile int* managedBufferOwner)
{
    int size = this_grid().size();
    int rank = this_grid().thread_rank(); // or this_multi_grid().grid_rank()

    auto t1 = clock64();
    
    for (int r = 0; r < REPETITIONS; r++) {
        p2pSendDeviceToHost(deviceSrcData, dataSize, 0, managedBuffer, managedBufferOwner);
        p2pSendHostToDevice(deviceDstData, dataSize, 0, managedBuffer, managedBufferOwner);
    }
    
    auto t2 = clock64();
    
    double totalTime = (t2 - t1) * 0.001 / peakClkKHz;
    double timePerSend = totalTime / REPETITIONS / 2;
    double bandwidth = dataSize / timePerSend;
    printf("dataSize = %d B, time = %lg us, bandwidth = %lg MB/s \n", int(dataSize), timePerSend * 1e6, bandwidth / 1e6);
}


int main() {
    int deviceCount = -1;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    assert(deviceCount >= 1);
    
    int peakClkKHz = -1;
    CUDA_CHECK(cudaDeviceGetAttribute(&peakClkKHz, cudaDevAttrClockRate, /*device = */0));
    assert(peakClkKHz > 0);
    

    
    for (size_t dataSize = 1; dataSize < (1 << 23); dataSize *= 2) {
        char* deviceSrcData = nullptr;
        CUDA_CHECK(cudaMalloc(&deviceSrcData, dataSize));
        
        char* deviceDstData = nullptr;
        CUDA_CHECK(cudaMalloc(&deviceDstData, dataSize));
        
        char* managedBuffer = nullptr;
        CUDA_CHECK(cudaMallocManaged(&managedBuffer, dataSize));

        int* managedBufferOwner = nullptr;
        CUDA_CHECK(cudaMallocManaged(&managedBufferOwner, sizeof(int)));
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
            (void*)&managedBufferOwner
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

        for (int r = 0; r < REPETITIONS; r++) {
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
        
        CUDA_CHECK(cudaFree(managedBuffer));
        CUDA_CHECK(cudaFree(managedBufferOwner));

        CUDA_CHECK(cudaFree(deviceDstData));
        CUDA_CHECK(cudaFree(deviceSrcData));
    }
    
    printf("Exit\n");
    
    return 0;
}
