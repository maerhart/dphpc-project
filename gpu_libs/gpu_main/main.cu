// Main function: entry point into GPU main()
// Only this file should use standard MPI implementation like OpenMPI or MPICH,
// functions on the GPU should use gpu_mpi instead.

#include <cuda.h>
//#include <mpi.h>

#include <stdio.h>

#include <string>
#include <vector>
#include <iostream>
#include <set>

#include <cxxopts.hpp>

#include "common.h"

#include "cuda_mpi.cuh"

#include "libc_processor.cuh"

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
int parseGPUMPIArgs(int argc, char** argv, 
    unsigned& blocksPerGrid, unsigned& threadsPerBlock, unsigned& stackSize, unsigned& heapSize) 
{
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
        ("g,blocksPerGrid", "Blocks per grid", cxxopts::value<unsigned>()->default_value("1"))
        ("b,threadsPerBlock", "Threads per block", cxxopts::value<unsigned>()->default_value("1"))
        ("s,stackSize", "Override stack size limit on GPU (bytes)", cxxopts::value<unsigned>()->default_value("1024"))
        ("p,heapSize", "Override heap size limit on GPU (bytes)", cxxopts::value<unsigned>()->default_value("8388608"))
        ("h,help", "Print help text")
        ;


    auto result = options.parse(gpumpi_argc, gpumpi_argv);

    if (result.count("help"))
    {
        std::cout << options.help() << std::endl;
        exit(0);
    }

    blocksPerGrid = result["blocksPerGrid"].as<unsigned>();
    threadsPerBlock = result["threadsPerBlock"].as<unsigned>();
    stackSize = result["stackSize"].as<unsigned>();
    heapSize = result["heapSize"].as<unsigned>();

    return trippleDashPosition;
}

extern __device__ int __gpu_main(int argc, char* argv[]);

__global__ void __gpu_main_caller(int argc, char* argv[],
                                    CudaMPI::SharedState* sharedState,
                                    CudaMPI::ThreadPrivateState::Context threadPrivateStateContext)
{
    CudaMPI::setSharedState(sharedState);
    CudaMPI::ThreadPrivateState::Holder threadPrivateStateHolder(threadPrivateStateContext);

    int returnValue = __gpu_main(argc, argv);
    if (returnValue != 0) {
        sharedState->returnValue = 1;
    }
}


// TODO: FIX. this is copypasted from converter.cpp, dangerous constant can be changed
const char* GPU_MPI_MAX_RANKS = "GPU_MPI_MAX_RANKS";
int getMaxRanks() {
    int res = 1024;

    char* maxRanks = getenv(GPU_MPI_MAX_RANKS);
    if (maxRanks) {
        res = atoi(maxRanks);
        if (res <= 0) {
            std::cerr << "ERROR: " << GPU_MPI_MAX_RANKS << " environment variable should contain number of ranks!\n";
            exit(1);
        }
    }

    return res;
}


int main(int argc, char* argv[]) {

    //MPI_CHECK(MPI_Init(&argc, &argv));

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

    unsigned blocksPerGrid = 0;
    unsigned threadsPerBlock = 0;
    unsigned stackSize = 0;
    unsigned heapSize = 0;

    int argcWithoutGPUMPI = parseGPUMPIArgs(argc, argv, blocksPerGrid, threadsPerBlock, stackSize, heapSize);

    if (blocksPerGrid * threadsPerBlock > getMaxRanks()) {
        printf("You trying to use more threads than supported by GPU MPI. You can increase the number of threads by\n");
        printf("overriding %s environment variable and recompiling the project.\n", GPU_MPI_MAX_RANKS);
        printf("WARNING! Without recompilation, the program is expected to crash!\n");
        exit(1);
    }

    // increase stack size
    CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, stackSize));
    // increase heap size
    CUDA_CHECK(cudaDeviceSetLimit(cudaLimitMallocHeapSize, heapSize));


    // convert the argv array into memory inside the an UM allocated buffer
    void* argvInUnifiedMemory = copyArgsToUnifiedMemory(argcWithoutGPUMPI,argv);

    // allocate memory for communication
    CudaMPI::SharedState::Context sharedStateContext;
    sharedStateContext.numThreads = blocksPerGrid * threadsPerBlock;
    sharedStateContext.recvListSize = 16;

    CudaMPI::SharedState::Holder sharedStateHolder(sharedStateContext);
    CudaMPI::SharedState* sharedState = sharedStateHolder.get();

    //create cuda streams for each device
    std::vector<cudaStream_t> cudaStreams(deviceCount);
    for(int i = 0; i < deviceCount; i++) {
        CUDA_CHECK(cudaSetDevice(i));
        CUDA_CHECK(cudaStreamCreate(&cudaStreams[i]));
    }

    std::vector<CudaMPI::ThreadPrivateState::Context> threadPrivateStateContext(deviceCount);
    for(int device = 0; device < deviceCount; device++) {
        threadPrivateStateContext[device].pendingBufferSize = 20;

        int peakClockKHz;
        CUDA_CHECK(cudaDeviceGetAttribute(&peakClockKHz, cudaDevAttrClockRate, device));
        threadPrivateStateContext[device].peakClockKHz = peakClockKHz;
    }

    // args passed into kernel function
    std::vector<std::vector<void*>> params(deviceCount);
    for(int i = 0; i < deviceCount; i++) {
        params[i] = {
            (void*)&argcWithoutGPUMPI,
            (void*)&argvInUnifiedMemory,
            (void*)&sharedState,
            (void*)&threadPrivateStateContext[i],
        };
    }

    std::vector<cudaLaunchParams> launchParamsList(deviceCount);
    for(int i = 0; i < deviceCount; i++) {
        launchParamsList[i].func = (void*) __gpu_main_caller;
        launchParamsList[i].gridDim = blocksPerGrid;
        launchParamsList[i].blockDim = threadsPerBlock;
        launchParamsList[i].args = params[i].data();
        launchParamsList[i].sharedMem = 0;
        launchParamsList[i].stream = cudaStreams[i];
    }

    cudaEvent_t kernelFinishEvent;
    CUDA_CHECK(cudaEventCreate(&kernelFinishEvent));

    std::cerr << "GPUMPI: Starting kernel!" << std::endl;
    // here we actually call __gpu_main
    //CUDA_CHECK(cudaLaunchCooperativeKernelMultiDevice(launchParamsList.data(), deviceCount));
    for(int i = 0; i < deviceCount; i++) {
        CUDA_CHECK(cudaLaunchKernel((void*)__gpu_main_caller, blocksPerGrid, threadsPerBlock, params[i].data(), 0, cudaStreams[i]));
    }
    CUDA_CHECK(cudaEventRecord(kernelFinishEvent));

    std::cerr << "GPUMPI: Processing messages from device threads" << std::endl;

    while (cudaEventQuery(kernelFinishEvent) == cudaErrorNotReady) {
        sharedState->deviceToHostCommunicator.processIncomingMessages([&](void* ptr, size_t size, int threadRank) {
            if (ptr == 0 && size == 0) {
                // nothing to do, this is notification that thread finished execution
            } else {
                process_gpu_libc(ptr, size);
            } 
        });
    }
    std::cerr << "GPUMPI: Kernel finished, stop processing messages from device threads" << std::endl;

    // make sure that everything is ok after kernel launch
    CUDA_CHECK(cudaEventQuery(kernelFinishEvent));

    std::cerr << "GPUMPI: Releasing resources" << std::endl;

    // release all resources

    CUDA_CHECK(cudaFree(argvInUnifiedMemory));

    //MPI_CHECK(MPI_Finalize());

    std::cerr << "GPUMPI: MPI finished!" << std::endl;

    return sharedState->returnValue;
}

