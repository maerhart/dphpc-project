#include <cuda.h>

#include <new>
#include <cstdio>
#include <chrono>
#include <cassert>

#include <unistd.h>

#define CHECK(expr) \
    do { if ((expr) != 0) { printf("ERROR %s:%d %s\n", __FILE__, __LINE__, #expr); abort(); } } while (0)

__forceinline__ __host__ __device__ void memfence() {
    #if __CUDA_ARCH__
        __threadfence_system();
    #else
        __sync_synchronize();
    #endif
}

__host__ __device__ void waitSecond() {
//    #if __CUDA_ARCH__
//        long long t1 = clock64();
//        while (clock64() - t1 < 1000000000) {}
//    #else
//        sleep(1);
//    #endif
}



struct SharedState {
    SharedState() {
    }

    __host__ __device__ int rank() {
        return threadIdx.x + blockDim.x * blockIdx.x;
    }

    __host__ __device__ int size() {
        return blockDim.x * gridDim.x;
    }

    // *** barrier between host and single device thread ***

    volatile bool hostBarrierReady = 0;
    volatile bool deviceBarrierReady = 0;

    __device__ void syncWithHost() {
        while (deviceBarrierReady) {} // wait previous entry
        deviceBarrierReady = true;
        while (!hostBarrierReady) {}
        hostBarrierReady = false;
    }

    void syncWithDeviceThread() {
        while (hostBarrierReady) {} // wait previous entry
        hostBarrierReady = true;
        while (!deviceBarrierReady) {}
        deviceBarrierReady = false;
    }

    // *** barrier between first threads of all blocks ***

    unsigned bi = 0;
    unsigned bo = 0;

    __device__ void syncBlocksMasters() {
        int numBlocks = gridDim.x;

        volatile unsigned* vbi = &bi;
        volatile unsigned* vbo = &bo;

        if (threadIdx.x == 0) {
            // wait other threads to exit from previous barrier invocation
            while (*vbo != 0) {}

            unsigned oldIn = atomicAdd_system(&bi, 1);

            // if we are last thread, reset out counter
            // and allow threads to pass barrier entry 
            if (oldIn == numBlocks - 1) {
                *vbo = numBlocks + 1;
                //memfence();
                *vbi += 1; // increase second time to numBlocks + 1
                //memfence();
            }
            
            // barrier entry
            while (*vbi != numBlocks + 1) {} 

            // if we are here, then all threads started exitting from barrier
            unsigned oldOut = atomicSub_system(&bo, 1);
            if (oldOut == 2) {
                *vbi = 0;
                //memfence();
                *vbo -= 1; // decrease second time to 0
                //memfence();
            }
        }
    }

    // barrier between all threads on device
    __device__ void deviceBarrier() {
        // Since synchtreads is not enough for correct barrier: 
        // one should be in the beginning and one at the end/
        __syncthreads();
        syncBlocksMasters();
        __syncthreads();
    }

    // barrier between all threads on device and host thread
    __host__ __device__ void hostDeviceBarrier() {
#ifdef __CUDA_ARCH__
        __syncthreads();
        if (rank() == 0) syncWithHost();
        syncBlocksMasters();
        if (rank() == 0) syncWithHost();
        __syncthreads();
#else
        // twice, it is not a bug!
        syncWithDeviceThread();
        syncWithDeviceThread();
#endif
    }


    volatile int value = 0;
};

using hrclock = std::chrono::high_resolution_clock;

template <typename T>
auto nanoseconds(T x) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(x);
}

const int iters = 10000;

__global__ void mykernel(SharedState* ss) {
    //int x;
    //if (ss->rank() == 0) {
    //    x = 2;
    //} else {
    //    x = 17;
    //}

    for (int i = 0; i < iters; i++) {
        ss->hostDeviceBarrier();

        //long long t1 = clock64();
        //while (clock64() - t1 < 1000000000) {}
        //printf("Device enters iteration %d\n", i);
        int rank = threadIdx.x + blockDim.x * blockIdx.x;
        if (i % 3 == 1 && rank == 1) {
            //printf("Device iteration %d\n", i);
            if (ss->value != 1) {
                printf("Device 1 ERROR!\n");
            }
            ss->value = 2;
            //memfence();
        }
        if (i % 3 == 2 && rank == 0) {
            //printf("Device iteration %d\n", i);
            if (ss->value != 2) {
                printf("Device 0 ERROR!\n");
            }
            ss->value = 0;
            //memfence();
        }

        //printf("Device exits iteration %d\n", i);

        if (ss->rank() == 0) {
            //x = (x + 1) % 25;
            if (i % (iters / 10) == 0) {
                printf("iter %d\n", i);
            }
        }
    }

}

int main() {
    SharedState* ss = nullptr;
    CHECK(cudaMallocManaged(&ss, sizeof(SharedState)));
    ss = new (ss) SharedState;

    mykernel<<<2,1>>>(ss);
    CHECK(cudaPeekAtLastError());

    for (int i = 0; i < iters; i++) {
        ss->hostDeviceBarrier();
        //waitSecond();
        //printf("Host enters iteration %d\n", i);
        if (i % 3 == 0) {
            //printf("Host iteration %d\n", i);
            if (ss->value != 0) {
                printf("Host ERROR!\n");
            }
            ss->value = 1;
            //memfence();
        }
        //printf("Host exits iteration %d\n", i);
    }

    CHECK(cudaDeviceSynchronize());

    ss->~SharedState();
    CHECK(cudaFree(ss));
}
