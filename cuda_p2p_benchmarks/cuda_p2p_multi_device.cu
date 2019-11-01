
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

__device__ void memcpy_volatile(volatile void *dst, volatile void *src, size_t n)
{
    volatile char *d = (char*) dst;
    volatile char *s = (char*) src;
    for (size_t i = 0; i < n; i++) {
        d[i] = s[i];
    }
}

__device__ void p2pSendDeviceToDevice(void* ptr, int size, int srcThread, int dstThread,
    volatile char* managedBuffer, volatile int* managedBufferOwner)
{
    int rank = this_multi_grid().thread_rank();
    if (rank != srcThread && rank != dstThread) return;

    if (rank == srcThread) {
        while (*managedBufferOwner != MANAGED_OWNER_NONE) {}

        memcpy_volatile(managedBuffer, ptr, size);
        __threadfence_system();
        *managedBufferOwner = dstThread;
    } else if (rank == dstThread) {
        while (*managedBufferOwner != dstThread) {}

        memcpy_volatile(ptr, managedBuffer, size);
        __threadfence_system();
        *managedBufferOwner = MANAGED_OWNER_NONE;
    }

}

#define REPETITIONS 10

__global__ void kernelBenchmarkMultiDevice(size_t dataSize, char* deviceSrcData, char* deviceDstData, int peakClkKHz,
                                          volatile char* managedBuffer, volatile int* managedBufferOwner)
{
    int size = this_multi_grid().size();
    int rank = this_multi_grid().thread_rank();


    char* data = nullptr;
    if (0 == rank) {
        data = deviceSrcData;
    } else if (1 == rank) {
        data = deviceDstData;
    }


    auto t1 = clock64();

    for (int r = 0; r < REPETITIONS; r++) {
        p2pSendDeviceToDevice(data, dataSize, 0, 1, managedBuffer, managedBufferOwner);
        p2pSendDeviceToDevice(data, dataSize, 1, 0, managedBuffer, managedBufferOwner);
    }

    auto t2 = clock64();

    if (rank == 0) {
        double totalTime = (t2 - t1) * 0.001 / peakClkKHz;
        double timePerSend = totalTime / REPETITIONS / 2;
        double bandwidth = dataSize / timePerSend;
        printf("dataSize = %d B, time = %lg us, bandwidth = %lg MB/s \n", int(dataSize), timePerSend * 1e6, bandwidth / 1e6);
    }
}


int main() {
    int deviceCount = -1;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount < 2) {
        printf("At least 2 devices are required\n");
        abort();
    }
    int devicesUsed = 2;

    int peakClkKHz = -1;
    CUDA_CHECK(cudaDeviceGetAttribute(&peakClkKHz, cudaDevAttrClockRate, /*device = */0));
    assert(peakClkKHz > 0);

    std::vector<cudaStream_t> cudaStreams(devicesUsed);
    for(int i = 0; i < devicesUsed; i++) {
        CUDA_CHECK(cudaSetDevice(i));
        CUDA_CHECK(cudaStreamCreate(&cudaStreams[i]));
    }

    std::vector<cudaLaunchParams> launchParamsList(devicesUsed);
    for(int i = 0; i < devicesUsed; i++) {
        launchParamsList[i].func = (void*) kernelBenchmarkMultiDevice;
        launchParamsList[i].gridDim = 1;
        launchParamsList[i].blockDim = 1;
        // launchParamsList[i].args = params; // it will be set later
        launchParamsList[i].sharedMem = 0;
        launchParamsList[i].stream = cudaStreams[i];
    }

    for (size_t dataSize = 1; dataSize < (1 << 23); dataSize *= 2) {
        char* deviceSrcData = nullptr;
        CUDA_CHECK(cudaSetDevice(0));
        CUDA_CHECK(cudaMalloc(&deviceSrcData, dataSize));

        char* deviceDstData = nullptr;
        CUDA_CHECK(cudaSetDevice(1));
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

        CUDA_CHECK(cudaSetDevice(0));
        CUDA_CHECK(cudaMemcpy(deviceSrcData, hostSrcData, dataSize, cudaMemcpyHostToDevice));

        void* params[] = {
            (void*)&dataSize,
            (void*)&deviceSrcData,
            (void*)&deviceDstData,
            (void*)&peakClkKHz,
            (void*)&managedBuffer,
            (void*)&managedBufferOwner
        };
        for(int i = 0; i < devicesUsed; i++) {
            launchParamsList[i].args = params;
        }

        CUDA_CHECK(cudaLaunchCooperativeKernelMultiDevice(launchParamsList.data(), devicesUsed));

        for(int i = 0; i < devicesUsed; i++) {
            CUDA_CHECK(cudaSetDevice(i));
            CUDA_CHECK(cudaPeekAtLastError());
        }

        for(int i = 0; i < devicesUsed; i++) {
            CUDA_CHECK(cudaSetDevice(i));
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        CUDA_CHECK(cudaSetDevice(1));
        CUDA_CHECK(cudaMemcpy(hostDstData, deviceDstData, dataSize, cudaMemcpyDeviceToHost));

        for (int i = 0; i < dataSize; i++) {
            if (hostDstData[i] != 17) {
                printf("Incorrect data!");
                abort();
            }
        }

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

