// Main function: entry point into GPU main()
// Only this file should use standard MPI implementation like OpenMPI or MPICH,
// functions on the GPU should use gpu_mpi instead.

#include <cuda.h>
#include <mpi.h>

#include <stdarg.h>

#include <stdio.h>

#include <string>
#include <vector>
#include <iostream>

#include "managed_memory.h"


#define CUDA_CHECK(expr) do {\
    cudaError_t err = (expr);\
    if (err != cudaSuccess) {\
        std::cerr << "CUDA ERROR: " << __FILE__ << ":" << __LINE__ << ": " << #expr << " <" << cudaGetErrorName(err) << "> " << cudaGetErrorString(err) << "\n"; \
        abort(); \
    }\
} while(0)

__global__ void mykernel() {
    for (int i = 0; i < 100; i++) {
        gManagedMemory.lock();
        ((int*)gManagedMemory.memory)[0] += 1;
        gManagedMemory.unlock();
    }
}

int main(int argc, char* argv[]) {
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

    void* params[] = {
        //(void*)&param1,
        //(void*)&param2
    };

    std::vector<cudaStream_t> cudaStreams(deviceCount);
    for(int i = 0; i < deviceCount; i++) {
        CUDA_CHECK(cudaSetDevice(i));
        CUDA_CHECK(cudaStreamCreate(&cudaStreams[i]));
    }

    std::vector<cudaLaunchParams> launchParamsList(deviceCount);
    for(int i = 0; i < deviceCount; i++) {
        launchParamsList[i].func = (void*) mykernel;
        launchParamsList[i].gridDim = 32;
        launchParamsList[i].blockDim = 16;
        launchParamsList[i].args = params;
        launchParamsList[i].sharedMem = 0;
        launchParamsList[i].stream = cudaStreams[i];
    }

    CUDA_CHECK(cudaLaunchCooperativeKernelMultiDevice(launchParamsList.data(), deviceCount));

    for (int i = 0; i < 10000; i++) {
        gManagedMemory.lock();
        ((int*)gManagedMemory.memory)[0] += 1;
        gManagedMemory.unlock();
    }


    for(int i = 0; i < deviceCount; i++) {
        CUDA_CHECK(cudaSetDevice(i));
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    printf("value: %d\n", ((int*)gManagedMemory.memory)[0]);

    printf("Exitting from main!\n");
    return 0;
}

