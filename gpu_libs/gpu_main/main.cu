// Main function: entry point into GPU main()
// Only this file should use standard MPI implementation like OpenMPI or MPICH,
// functions on the GPU should use gpu_mpi instead.

#include <cuda.h>
#include <mpi.h>

#include <stdio.h>

#include <string>
#include <vector>
#include <iostream>

#define $ std::cerr << "STILL ALIVE: " << __LINE__ << std::endl;


#define CUDA_CHECK(expr) do {\
    cudaError_t err = (expr);\
    if (err != cudaSuccess) {\
        std::cerr << "CUDA ERROR: " << __FILE__ << ":" << __LINE__ << ": " << #expr << " <" << cudaGetErrorName(err) << "> " << cudaGetErrorString(err) << "\n"; \
        abort(); \
    }\
} while(0)

#define MPI_CHECK(expr) \
    if ((expr) != MPI_SUCCESS) { \
        std::cerr << "MPI ERROR: " << __FILE__ << ":" << __LINE__ << ": " << #expr << "\n"; \
        abort(); \
    }


void* copyArgsToUnifiedMemory(int argc, char** argv) {
    // argv is a set of "pointers" to "strings"
    int stringsSize = 0;
    for (int i = 0; i < argc; i++) {
        stringsSize += (strlen(argv[i]) + 1);
    }
    void* argsInUnifiedMemory = NULL;
    int pointersSize = (argc + 1) * sizeof(char*);
    CUDA_CHECK(cudaMallocManaged(&argsInUnifiedMemory, pointersSize + stringsSize));
    char** pointers = (char**) argsInUnifiedMemory;
    char* strings = (char*)&pointers[argc+1];
    char* current_string = strings;
    for (int i = 0; i < argc; i++) {
        pointers[i] = current_string;
        strcpy(current_string, argv[i]);
        current_string += (strlen(argv[i]) + 1);
    }
    pointers[argc] = NULL;
    return argsInUnifiedMemory;
}

extern __global__ void __gpu_main_kernel(int argc, char* argv[]);


int main(int argc, char* argv[]) {

    MPI_CHECK(MPI_Init(&argc, &argv));

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

    // convert the argv array into memory inside the an UM allocated buffer
    void * argsInUnifiedMemory = copyArgsToUnifiedMemory(argc,argv);

    void* params[2] = {
        (void*)&argc,
        (void*)&argsInUnifiedMemory
    };

    //create cuda streams for each device
    std::vector<cudaStream_t> cudaStreams(deviceCount);
    for(int i = 0; i < deviceCount; i++) {
        CUDA_CHECK(cudaSetDevice(i));
        CUDA_CHECK(cudaStreamCreate(&cudaStreams[i]));
    }

    std::vector<cudaLaunchParams> launchParamsList(deviceCount);
    for(int i = 0; i < deviceCount; i++) {
        launchParamsList[i].func = (void*) __gpu_main_kernel;
        launchParamsList[i].gridDim = 1;
        launchParamsList[i].blockDim = 1;
        launchParamsList[i].args = params;
        launchParamsList[i].sharedMem = 0;
        launchParamsList[i].stream = cudaStreams[i];
    }

    std::cerr << "Starting kernel!" << std::endl;
    // here we actually call __gpu_main
    CUDA_CHECK(cudaLaunchCooperativeKernelMultiDevice(launchParamsList.data(), deviceCount));
    std::cerr << "Finishing kernel!" << std::endl;

    for(int i = 0; i < deviceCount; i++) {
        CUDA_CHECK(cudaSetDevice(i));
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    std::cerr << "Synchronized!" << std::endl;

    MPI_CHECK(MPI_Finalize());

    std::cerr << "MPI finished!" << std::endl;
}

