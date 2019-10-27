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

struct BlockMessageBuffer {
    __device__ void init(const thread_group& thread_group) {
        mThreadGroup = &thread_group;
        if (mThreadGroup->thread_rank() == 0) {
            srcThreadOwnsBuffer = true;
        }
        mThreadGroup->sync();
    }
    
    __device__ void sendrecv(void* ptr, int size, int srcThread, int dstThread) {
        int bytesWritten = 0;
        
        while (bytesWritten < size) {
            /*
             * It is important to process both sender and receiver inside the same while
             * loop body. If both threads inside the same warp, it allows to execute
             * them simultaneously without deadlock.
             */
            
            int bytesLeft = size - bytesWritten;
            int bytesToWrite = bytesLeft > SharedBufferSize ? SharedBufferSize : bytesLeft;
            if (srcThreadOwnsBuffer) {
                if (srcThread == mThreadGroup->thread_rank()) {
                    memcpy((void*)sharedBuffer, ((char*)ptr) + bytesWritten, bytesToWrite);
                    bytesWritten += bytesToWrite;
                    __threadfence_block();
                    srcThreadOwnsBuffer = false;
                }
            } else {
                if (dstThread == mThreadGroup->thread_rank()) {
                    memcpy(((char*)ptr) + bytesWritten, (void*)sharedBuffer, bytesToWrite);
                    bytesWritten += bytesToWrite;
                    __threadfence_block();
                    srcThreadOwnsBuffer = true;
                }
            }
        }
    }

    enum { SharedBufferSize = 128 };

private:
    
    const thread_group* mThreadGroup;
    volatile char sharedBuffer[SharedBufferSize];
    volatile bool srcThreadOwnsBuffer;
};

struct MemoryBuffer {
    void* buffer;
    size_t srcThreadOwnsBuffer;
};

struct MemoryBuffers {
    __device__ void init() {

        int rank = this_grid().thread_rank();
        int size = this_grid().size();
        if (rank == 0) {
            buffers = (MemoryBuffer*) malloc(sizeof(*buffers) * size);
            assert(buffers);
        }
        this_grid().sync();
        buffers[rank].buffer = nullptr;
        buffers[rank].srcThreadOwnsBuffer = true;
        this_grid().sync();

    }
    __device__ void destroy() {
        this_grid().sync();
        if (this_grid().thread_rank() == 0) {
            free((void*)buffers);
        }
    }
    volatile MemoryBuffer* buffers;
};

__device__ MemoryBuffers gDeviceMessageBuffers;

__device__ void device_sendrecv(void* ptr, int size, int srcThread, int dstThread) {
    // TODO: to avoid allocation of large chunks of memory, it is possible to use __isGlobal() check and reuse ptr
    
    int rank = this_grid().thread_rank();
    
    if (rank != srcThread && rank != dstThread) return;
    
    bool done = false;
    
    void* memChunk = nullptr;
    if (rank == srcThread) {
        memChunk = malloc(size);
        assert(memChunk);
    }
    
    volatile MemoryBuffer* buf = &gDeviceMessageBuffers.buffers[srcThread];
    
    while (!done) {
        if (rank == srcThread && buf->srcThreadOwnsBuffer) {

            buf->buffer = memChunk;
            memcpy(buf->buffer, ptr, size);
            
            __threadfence();
            
            buf->srcThreadOwnsBuffer = false;
            
            done = true;
        } else if (rank == dstThread && !buf->srcThreadOwnsBuffer) {
            memcpy(ptr, buf->buffer, size);
            
            free(buf->buffer);
            buf->buffer = nullptr;
            
            __threadfence();
            
            buf->srcThreadOwnsBuffer = true;
            
            done = true;
        }
    }
}

__managed__ int peakClkKHz;

__device__ double cudaTime() {
    return clock64() * 0.001 / peakClkKHz;
}

__global__ void mykernel_warp() {
    thread_block g = this_thread_block();
    
    const int maxDataSize = 1 << 20;
    int* data = (int*)malloc(maxDataSize * sizeof(*data));
    assert(data);
    
    if (g.thread_rank() == 0) {
        for (int i = 0; i < maxDataSize; i++) {
            data[i] = i;
        }
        printf("Rank 0 thread has written the data\n");
    }
    
    __shared__ BlockMessageBuffer bmb;
    bmb.init(g);
        
    double time = cudaTime();
        
    bmb.sendrecv(data, sizeof(*data) * maxDataSize, 0, 1);
        
    time = cudaTime() - time;
    
    if (g.thread_rank() == 1) {
        for (int i = 0; i < maxDataSize; i++) {
            assert(data[i] == i);
        }
        printf("Rank 1 thread has checked the data\n");
        printf("Time: %lg s\n", time);
    }
    
    free(data);
}

__global__ void mykernel_block() {
    coalesced_group warp = coalesced_threads();
    if (warp.thread_rank() != 0) return;
    
    printf("mykernel_block: %d\n", this_thread_block().thread_rank());

    const int maxDataSize = 1 << 20;
    int* data = (int*)malloc(maxDataSize * sizeof(*data));
    assert(data);
    
    if (this_thread_block().thread_rank() == 0) {
        for (int i = 0; i < maxDataSize; i++) {
            data[i] = i;
        }
        printf("Rank 0 thread has written the data\n");
    }
    
    __shared__ BlockMessageBuffer bmb;
    
    bmb.init(this_thread_block());
        
    double time = cudaTime();
        
    bmb.sendrecv(data, sizeof(*data) * maxDataSize, 0, 32);
        
    time = cudaTime() - time;
    
    if (this_thread_block().thread_rank() == 32) {
        for (int i = 0; i < maxDataSize; i++) {
            assert(data[i] == i);
        }
        printf("Rank 32 thread has checked the data\n");
        printf("Time: %lg s\n", time);
    }
    
    free(data);
}

__global__ void mykernel_device() {
    const int maxDataSize = 1 << 20;
    int* data = (int*)malloc(maxDataSize * sizeof(*data));
    assert(data);
    
    if (this_grid().thread_rank() == 0) {
        for (int i = 0; i < maxDataSize; i++) {
            data[i] = i;
        }
        printf("Rank 0 thread has written the data\n");
    }
    
    gDeviceMessageBuffers.init();
            
    double time = cudaTime();
        
    device_sendrecv(data, sizeof(*data) * maxDataSize, 0, 1);
    
    time = cudaTime() - time;
    
    if (this_grid().thread_rank() == 1) {
        for (int i = 0; i < maxDataSize; i++) {
            assert(data[i] == i);
        }
        printf("Rank 1 thread has checked the data\n");
        printf("Time: %lg s\n", time);
    }
    
    gDeviceMessageBuffers.destroy();
    
    free(data);
}

#define ManagedBufferSize 128
__managed__ volatile char gManagedBuffer[ManagedBufferSize];
__managed__ volatile bool gManagedSrcThreadOwnsBuffer = true;

__device__ void multidevice_sendrecv(void* ptr, int size, int srcThread, int dstThread) {
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

__global__ void mykernel_host() {
    int size = this_multi_grid().size();
    int rank = this_multi_grid().thread_rank(); // or this_multi_grid().grid_rank()

    printf("mykernel_host, rank = %d\n", rank);
    
    const int maxDataSize = 1 << 20;
    int* data = (int*)malloc(maxDataSize * sizeof(*data));
    assert(data);
    
    if (rank == 0) {
        for (int i = 0; i < maxDataSize; i++) {
            data[i] = i;
        }
        printf("Rank 0 thread has written the data\n");
    }

    double time = cudaTime();

    multidevice_sendrecv(data, sizeof(*data) * maxDataSize, 0, 1);

    time = cudaTime() - time;
    
    if (rank == 1) {
        for (int i = 0; i < maxDataSize; i++) {
            assert(data[i] == i);
        }
        printf("Rank 1 thread has checked the data\n");
        printf("Time: %lg s\n", time);
    }
    
    free(data);
}

int main() {
    int deviceCount = -1;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    assert(deviceCount >= 1);

    // maximum malloc allocatable memory on device
    CUDA_CHECK(cudaDeviceSetLimit(cudaLimitMallocHeapSize, 1u << 29));
    
    int device = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&peakClkKHz, cudaDevAttrClockRate, device));
    
    mykernel_warp<<<1, 2>>>();

    CUDA_CHECK(cudaDeviceSynchronize());

    mykernel_block<<<1, 64>>>();
    
    CUDA_CHECK(cudaDeviceSynchronize());

    void* params[] = {};
    
    CUDA_CHECK(cudaLaunchCooperativeKernel(
        /*const T *func*/(void*)mykernel_device,
        /*dim3 gridDim*/dim3(2),
        /*dim3 blockDim*/dim3(1),
        /*void **args*/params
        /*size_t sharedMem = 0,
          cudaStream_t stream = 0*/
        ));
    
    CUDA_CHECK(cudaDeviceSynchronize());
    
    printf("deviceCount = %d\n", deviceCount);
    if (deviceCount >= 2) {
        int devicesUsed = deviceCount;
        printf("deviceUsed = %d\n", devicesUsed);
        
        void* params[] = {};

        std::vector<cudaStream_t> cudaStreams(devicesUsed);
        for(int i = 0; i < devicesUsed; i++) {
            CUDA_CHECK(cudaSetDevice(i));
            CUDA_CHECK(cudaStreamCreate(&cudaStreams[i]));
        }

        std::vector<cudaLaunchParams> launchParamsList(devicesUsed);
        for(int i = 0; i < devicesUsed; i++) {
            launchParamsList[i].func = (void*) mykernel_host;
            launchParamsList[i].gridDim = 1;
            launchParamsList[i].blockDim = 1;
            launchParamsList[i].args = params;
            launchParamsList[i].sharedMem = 0;
            launchParamsList[i].stream = cudaStreams[i];
        }

        CUDA_CHECK(cudaLaunchCooperativeKernelMultiDevice(launchParamsList.data(), devicesUsed));

        for(int i = 0; i < devicesUsed; i++) {
            CUDA_CHECK(cudaSetDevice(i));
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }

    printf("Exitting from main!\n");
    return 0;
}

