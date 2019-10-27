
#include <cuda.h>
#include <cooperative_groups.h>

#include <iostream>
#include <vector>
#include <cassert>

using namespace cooperative_groups;
using namespace std;

#define CUDA_CHECK(expr) do {\
    cudaError_t err = (expr);\
    if (err != cudaSuccess) {\
        std::cerr << "CUDA ERROR: " << __FILE__ << ":" << __LINE__ << ": " << #expr << " <" << cudaGetErrorName(err) << "> " << cudaGetErrorString(err) << "\n"; \
        abort(); \
    }\
} while(0)

__host__ __device__ void p2pSendHostDevice(void* ptr, int size, int srcThread, int dstThread) {
    int rank = this_multi_grid().thread_rank();
    
    if (rank != srcThread && rank != dstThread) return;
    
    int bytesWritten = 0;
        
    while (bytesWritten < size) {
        int bytesLeft = size - bytesWritten;
        int bytesToWrite = bytesLeft > ManagedBufferSize ? ManagedBufferSize : bytesLeft;
        if (gManagedSrcThreadOwnsBuffer) {
            if (srcThread == rank) {
                memcpy((void*)gManagedBuffer, ((char*)ptr) + bytesWritten, bytesToWrite);
                bytesWritten += bytesToWrite;
                __threadfence_system();
                gManagedSrcThreadOwnsBuffer = false;
            }
        } else {
            if (dstThread == rank) {
                memcpy(((char*)ptr) + bytesWritten, (void*)gManagedBuffer, bytesToWrite);
                bytesWritten += bytesToWrite;
                __threadfence_system();
                gManagedSrcThreadOwnsBuffer = true;
            }
        }
    }
}

__global__ void kernelBenchmarkHostDevice(size_t dataSize, char* deviceSrcData, char* deviceDstData, int peakClkKHz, 
                                          char* volatile managedBuffer, bool* volatile managedBufferOwnedByHost) 
{
    int size = this_grid().size();
    int rank = this_grid().thread_rank(); // or this_multi_grid().grid_rank()

    int repetitions = 10;
    
    void* data = nullptr;
    if (rank == 0) {
        data = deviceSrcData;
    } else {
        data = deviceDstData;
    }
    
    auto t1 = clock64();
    
    for (int r = 0; r < repetitions; r++) {
        p2pSendDevice(data, dataSize, /*srcThread*/ 0, /*dstThread*/ 1);
        p2pSendDevice(data, dataSize, /*srcThread*/ 1, /*dstThread*/ 0);
    }
    
    auto t2 = clock64();
    
    if (this_grid().thread_rank() == 0) {
        double totalTime = (t2 - t1) * 0.001 / peakClkKHz;
        double timePerSend = totalTime / repetitions / 2;
        double bandwidth = dataSize / timePerSend;
        printf("dataSize = %d B, time = %lg us, bandwidth = %lg Mb/s \n", int(dataSize), timePerSend * 1e6, bandwidth / 1e6);
    }
}


int main() {
    int deviceCount = -1;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    assert(deviceCount >= 1);
    
    int peakClkKHz = -1;
    CUDA_CHECK(cudaDeviceGetAttribute(&peakClkKHz, cudaDevAttrClockRate, /*device = */0));
    assert(peakClkKHz > 0);
    
    __device__ void* volatile managedBufferData;
    
    for (size_t dataSize = 1; dataSize < (1 << 23); dataSize *= 2) {
        char* deviceSrcData = nullptr;
        CUDA_CHECK(cudaMalloc(&deviceSrcData, dataSize));
        
        char* deviceDstData = nullptr;
        CUDA_CHECK(cudaMalloc(&deviceDstData, dataSize));
        
        char* managedBuffer = nullptr;
        CUDA_CHECK(cudaMallocManaged(&managedBuffer, dataSize));
        
        bool* managedBufferOwnedByHost = nullptr;
        CUDA_CHECK(cudaMallocManaged(&managedBufferOwnedByHost, sizeof(bool)));
        
        char* hostSrcData = (char*) calloc(dataSize, 1);
        assert(hostSrcData);
        char* hostDstData = (char*) calloc(dataSize, 1);
        assert(hostDstData);
        
        for (int i = 0; i < dataSize; i++) hostSrcData[i] = 17;
        
        CUDA_CHECK(cudaMemcpy(deviceSrcData, hostSrcData, dataSize, cudaMemcpyHostToDevice));
        
        dim3 blocksPerGrid = 2;
        dim3 threadsPerBlock = 1;
        
        void* params[] = {
            (void*)&dataSize,
            (void*)&deviceSrcData,
            (void*)&deviceDstData,
            (void*)&peakClkKHz
        };
        
        CUDA_CHECK(cudaLaunchCooperativeKernel(
            /*const T *func*/(void*) kernelBenchmarkHostDevice,
            /*dim3 gridDim*/blocksPerGrid,
            /*dim3 blockDim*/threadsPerBlock,
            /*void **args*/params
            /*size_t sharedMem = 0,
            cudaStream_t stream = 0*/
            ));
        
        CUDA_CHECK(cudaPeekAtLastError());
        
        while () {
        
        CUDA_CHECK(cudaDeviceSynchronize());
        
        CUDA_CHECK(cudaMemcpy(hostDstData, deviceDstData, dataSize, cudaMemcpyDeviceToHost));
        
        for (int i = 0; i < dataSize; i++) {
            if (hostDstData[i] != 17) {
                printf("Incorrect data!");
                abort();
            }
        }
        
        free(hostDstData);
        free(hostSrcData);
        
        CUDA_CHECK(cudaFree(deviceDstData));
        CUDA_CHECK(cudaFree(deviceSrcData));
    }
    
    printf("Exit\n");
    
    return 0;
}
