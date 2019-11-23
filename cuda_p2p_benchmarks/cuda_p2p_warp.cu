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


extern __shared__ void* sharedBuffer[];
#define SHARED_BUFFER_OWNER (((int*)sharedBuffer)[0])
#define SHARED_BUFFER_DATA (void*)(&((int*)sharedBuffer)[1])

__device__ void p2pSendWarp(void* data, size_t dataSize, int srcThread, int dstThread) {
    int rank = this_thread_block().thread_rank();
    
    if (srcThread != rank && dstThread != rank) return;

    bool done = false;
    while (!done) {
        if (srcThread == rank) {
            if (atomicCAS(&SHARED_BUFFER_OWNER, -1, srcThread) == srcThread) {
                memcpy(SHARED_BUFFER_DATA, data, dataSize);
                // next line prevents other threads from using data before
                // current thread entirely copied it
                __threadfence_block();
                SHARED_BUFFER_OWNER = dstThread;
                // current thread allowed to go out of loop
                // only if other threads notified that he changed ownership
                __threadfence_block();
                done = true;
            }
        } else {
            if (SHARED_BUFFER_OWNER == dstThread) {
                memcpy(data, SHARED_BUFFER_DATA, dataSize);
                __threadfence_block();
                SHARED_BUFFER_OWNER = -1;
                __threadfence_block();
                done = true;
            }
        }
    }
}

__global__ void kernelBenchmarkWarp(size_t dataSize, char* deviceSrcData, char* deviceDstData, int peakClkKHz) {
        

    int rank = this_thread_block().thread_rank();
    
    if (rank >= 2) return;
    
    if (rank == 0) {
        SHARED_BUFFER_OWNER = 0;
    }
    this_thread_block().sync();
    void* data = nullptr;
    if (rank == 0) {
        data = deviceSrcData;
    } else {
        data = deviceDstData;
    }
    
    int repetitions = 100;

    double E_T = 0;
    double E_T2 = 0;

    double E_BW = 0;
    double E_BW2 = 0;

    int dataPerSendRecv = 2 * dataSize;
    
    for (int r = 0; r < repetitions; r++) {
        auto t1 = clock64();
        p2pSendWarp(data, dataSize, /*srcThread*/ 0, /*dstThread*/ 1);
        p2pSendWarp(data, dataSize, /*srcThread*/ 1, /*dstThread*/ 0);
        auto t2 = clock64();

        double dt = (t2 - t1) * 0.001 / peakClkKHz;
        double bw = dataPerSendRecv / dt;

        E_T += dt;
        E_T2 += dt * dt;

        E_BW += bw;
        E_BW2 += bw * bw;
    }
    
    E_T /= repetitions;
    E_T2 /= repetitions;

    E_BW /= repetitions;
    E_BW2 /= repetitions;

    double timePerSendRecv = E_T;
    double timePerSendRecvErr = sqrt(E_T2 - E_T * E_T);
    double bandwidth = E_BW;
    double bandwidthErr = sqrt(E_BW2 - E_BW * E_BW);

    if (this_thread_block().thread_rank() == 0) {
        printf("dataSize = %d B, time = %lg ( +- %lg ) us, bandwidth = %lg ( +- %lg ) MB/s \n",
               int(dataSize), timePerSendRecv * 1e6, timePerSendRecvErr * 1e6, bandwidth / 1e6, bandwidthErr / 1e6);
    }
}

int main() {
    int deviceCount = -1;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    assert(deviceCount >= 1);
    
    int peakClkKHz = -1;
    CUDA_CHECK(cudaDeviceGetAttribute(&peakClkKHz, cudaDevAttrClockRate, /*device = */0));
    assert(peakClkKHz > 0);
    
    int sharedMemPerBlock = -1;
    CUDA_CHECK(cudaDeviceGetAttribute(&sharedMemPerBlock, cudaDevAttrMaxSharedMemoryPerBlock, /*device = */0));
    // there is also sharedMemPerMultiprocessor
    assert(sharedMemPerBlock > 0);
    
    
    
    for (size_t dataSize = 1; dataSize < sharedMemPerBlock - sizeof(int); dataSize *= 2) {
        
        char* deviceSrcData = nullptr;
        CUDA_CHECK(cudaMalloc(&deviceSrcData, dataSize));
        
        char* deviceDstData = nullptr;
        CUDA_CHECK(cudaMalloc(&deviceDstData, dataSize));
        
        char* hostSrcData = (char*) malloc(dataSize);
        assert(hostSrcData);
        char* hostDstData = (char*) malloc(dataSize);
        assert(hostDstData);
        
        for (int i = 0; i < dataSize; i++) hostSrcData[i] = 17;
        
        CUDA_CHECK(cudaMemcpy(deviceSrcData, hostSrcData, dataSize, cudaMemcpyHostToDevice));
        
        dim3 blocksPerGrid = 1;
        dim3 threadsPerBlock = 2;
        size_t sharedBufferSize = dataSize + sizeof(int);
        kernelBenchmarkWarp<<<blocksPerGrid, threadsPerBlock, sharedBufferSize>>>(dataSize, deviceSrcData, deviceDstData, peakClkKHz);
        
        CUDA_CHECK(cudaPeekAtLastError());
        
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
