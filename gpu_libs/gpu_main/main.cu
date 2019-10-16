// Main function: entry point into GPU main()
// Only this file should use standard MPI implementation like OpenMPI or MPICH,
// functions on the GPU should use gpu_mpi instead.

#include <cuda.h>
#include <mpi.h>

#include <stdio.h>

#include <string>
#include <vector>
#include <iostream>

#include <cxxopts.hpp>

#include "hostdevicecommunicator.cuh"

#include "common.h"

void* copyArgsToUnifiedMemory(int argc, char** argv) {
    // argv is a set of "pointers" to "strings"
    int stringsSize = 0;
    for (int i = 0; i < argc; i++) {
        stringsSize += (strlen(argv[i]) + 1);
    }
    char* argvInUnifiedMemory = NULL;
    int pointersSize = (argc + 1) * sizeof(char*);
    CUDA_CHECK(cudaMallocManaged(&argvInUnifiedMemory, pointersSize + stringsSize));
    char** pointers = (char**) argvInUnifiedMemory;
    char* strings = (char*)&pointers[argc+1];
    char* current_string = strings;
    for (int i = 0; i < argc; i++) {
        pointers[i] = current_string;
        strcpy(current_string, argv[i]);
        current_string += (strlen(argv[i]) + 1);
    }
    pointers[argc] = NULL;
    return argvInUnifiedMemory;
}

/*
 * Parse args related to gpumpi: everything after "---gpumpi"
 * Return new argc: everything after "---gpumpi"
 */
int parseGPUMPIArgs(int argc, char** argv, int& blocksPerGrid, int& threadsPerBlock) {
    int trippleDashPosition = -1;
    for (int i = 0; i < argc; i++) {
        if (strcmp(argv[i], "---gpumpi") == 0) {
            trippleDashPosition = i;
        }
    }
    if (trippleDashPosition == -1) {
        // no tripple dash, say about it and exit
        std::cerr << "You should specify gpumpi related options after '---gpumpi'" << std::endl;
        std::exit(1);
    }

    char** gpumpi_argv = argv + trippleDashPosition;
    int gpumpi_argc = argc - trippleDashPosition;

    cxxopts::Options options("GPU MPI", "GPU MPI");

    options.add_options()
        ("g,blocksPerGrid", "Enable debugging", cxxopts::value<int>())
        ("b,threadsPerBlock", "File name", cxxopts::value<int>())
        ;

    auto result = options.parse(gpumpi_argc, gpumpi_argv);
    blocksPerGrid = result["blocksPerGrid"].as<int>();
    threadsPerBlock = result["threadsPerBlock"].as<int>();

    return trippleDashPosition;
}

extern __global__ void __gpu_main_kernel(int argc, char* argv[]);

int main(int argc, char* argv[]) {

    MPI_CHECK(MPI_Init(&argc, &argv));

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

    int blocksPerGrid = -1;
    int threadsPerBlock = -1;

    int argcWithoutGPUMPI = parseGPUMPIArgs(argc, argv, blocksPerGrid, threadsPerBlock);

    // convert the argv array into memory inside the an UM allocated buffer
    void* argvInUnifiedMemory = copyArgsToUnifiedMemory(argcWithoutGPUMPI,argv);

    // allocate memory for host-thread communication
    gHostDeviceCommunicator.init(blocksPerGrid, threadsPerBlock);

    // args passed into kernel function
    void* params[] = {
        (void*)&argcWithoutGPUMPI,
        (void*)&argvInUnifiedMemory
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
        launchParamsList[i].gridDim = blocksPerGrid;
        launchParamsList[i].blockDim = threadsPerBlock;
        launchParamsList[i].args = params;
        launchParamsList[i].sharedMem = 0;
        launchParamsList[i].stream = cudaStreams[i];
    }

    std::cerr << "Starting kernel!" << std::endl;
    // here we actually call __gpu_main
    CUDA_CHECK(cudaLaunchCooperativeKernelMultiDevice(launchParamsList.data(), deviceCount));
    std::cerr << "Finishing kernel!" << std::endl;

    gHostDeviceCommunicator.processMessages();

    // wait while all devices are finishing computations
    for(int i = 0; i < deviceCount; i++) {
        CUDA_CHECK(cudaSetDevice(i));
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    std::cerr << "Synchronized!" << std::endl;

    // release all resources

    CUDA_CHECK(cudaFree(argvInUnifiedMemory));

    gHostDeviceCommunicator.destroy();

    MPI_CHECK(MPI_Finalize());

    std::cerr << "MPI finished!" << std::endl;
}

