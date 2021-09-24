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
#include "assert.cuh"
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
    size_t& numProcs, unsigned& blocksPerGrid, unsigned& threadsPerBlock,
    size_t& stackSize, size_t& heapSize, unsigned& pendingBufferSize) 
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
        ("n,numProcs", "Total number of processes", cxxopts::value<size_t>()->default_value("0"))
        ("g,blocksPerGrid", "Blocks per grid", cxxopts::value<unsigned>()->default_value("0"))
        ("b,threadsPerBlock", "Threads per block", cxxopts::value<unsigned>()->default_value("0"))
        ("s,stackSize", "Override stack size limit on GPU (bytes)", cxxopts::value<size_t>()->default_value("1024"))
        ("p,heapSize", "Override heap size limit on GPU (bytes)", cxxopts::value<size_t>()->default_value("0"))
        ("e,pendingBufferSize", "Override size of thread-local buffer of pending messages", cxxopts::value<unsigned>()->default_value("1024"))
        ("h,help", "Print help text")
        ;


    auto result = options.parse(gpumpi_argc, gpumpi_argv);

    if (result.count("help"))
    {
        std::cout << options.help() << std::endl;
        exit(0);
    }

    numProcs = result["numProcs"].as<size_t>();
    blocksPerGrid = result["blocksPerGrid"].as<unsigned>();
    threadsPerBlock = result["threadsPerBlock"].as<unsigned>();
    stackSize = result["stackSize"].as<size_t>();
    heapSize = result["heapSize"].as<size_t>();
    pendingBufferSize = result["pendingBufferSize"].as<unsigned>();

    return trippleDashPosition;
}

extern __device__ int __gpu_main(int argc, char* argv[]);

__global__ void __gpu_main_caller(int argc, char* argv[],
                                    CudaMPI::SharedState* sharedState,
                                    CudaMPI::ThreadPrivateState::Context threadPrivateStateContext)
{
    // finish extra threads launched because of the requirement to be a factor of block size
    if (sharedState->gridRank() > sharedState->activeGridSize()) return;

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

    size_t numProcs = 0;
    unsigned blocksPerGrid = 0;
    unsigned threadsPerBlock = 0;
    size_t stackSize = 0;
    size_t heapSize = 0;
    unsigned pendingBufferSize = 0;

    int argcWithoutGPUMPI = parseGPUMPIArgs(argc, argv, numProcs, blocksPerGrid, threadsPerBlock, stackSize, heapSize, pendingBufferSize);

    int computeCapabilityMajor = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&computeCapabilityMajor, cudaDevAttrComputeCapabilityMajor, /* device */ 0));
    bool hasIndependentThreadScheduling = computeCapabilityMajor >= 7;

    if (!hasIndependentThreadScheduling && threadsPerBlock == 0) {
        threadsPerBlock = 1;
    }
    if (!hasIndependentThreadScheduling && threadsPerBlock > 1) {
        printf("GPUMPI: GPU doesn't support independent thread scheduling. The max threads per block is limited to 1, while %u requested.\n", threadsPerBlock);
        printf("GPUMPI: Exitting...\n");
        exit(1);
    }

    // try to intelligently pick the best allocation based on user requirements
    if (numProcs == 0) {
        if (blocksPerGrid == 0) blocksPerGrid = 1;
        if (threadsPerBlock == 0) threadsPerBlock = 1;
        numProcs = blocksPerGrid * threadsPerBlock;
    } else {
        if (threadsPerBlock == 0) {
            int minGridSize = 0;
            int blockSize = 0;
            CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, __gpu_main_caller, /*dynamicSMemSize*/ 0, /*blockSizeLimit*/ 0));
            (void)minGridSize; // unused
            threadsPerBlock = blockSize;
        }

        if (blocksPerGrid == 0) {
            blocksPerGrid = (numProcs / threadsPerBlock) + ((numProcs % threadsPerBlock) ? 1 : 0);
        }
    }

    if (numProcs > blocksPerGrid * threadsPerBlock) {
        printf("GPUMPI: Requested number of processes %zu can't be allocated on %u blocks and %u threads\n", numProcs, blocksPerGrid, threadsPerBlock);
        printf("GPUMPI: Exitting...\n");
        exit(1);
    }
    printf("GPUMPI: Using %zu mpi processes with %u blocks and %u threads on GPU\n", numProcs, blocksPerGrid, threadsPerBlock);

    if (numProcs > getMaxRanks()) {
        printf("GPUMPI: You trying to use more threads than supported by GPU MPI."
               "You can increase the number of threads by overriding %s environment variable and recompiling the project.\n", GPU_MPI_MAX_RANKS);
        printf("GPUMPI: Exitting...\n");
        exit(1);
    }

    int blocksPerMP = -1;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerMP, __gpu_main_caller, threadsPerBlock, /*sharedMem*/ 0));
    printf("GPUMPI: Max active blocks per multiprocessor %d (for %d thread(s) per block)\n", blocksPerMP, threadsPerBlock);

    int multiProcessorCount = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&multiProcessorCount, cudaDevAttrMultiProcessorCount, /*device*/ 0));
    
    // According to documentation for cudaLaunchCooperativeKernel this is the max number of blocks we can run.
    // Even while we are not using cudaLaunchCooperativeKernel, we have the same requirement
    // to support grid-wide synchronization. It is only possible if all threads are active.
    // When limit on the number of blocks is exceed, blocks scheduled sequentially, so
    // global barrier will cause the deadlock as
    // block 0 can't finish before block N is started and block N can't start before block 0 is finished.
    int maxBlocks = blocksPerMP * multiProcessorCount;
    printf("GPUMPI: Max number of blocks is %d (for %d thread(s) per block)\n", maxBlocks, threadsPerBlock);
    
    if (blocksPerGrid > maxBlocks) {
        printf("GPUMPI: The requested number of blocks (%d) exceeds the maximum number of blocks (%d) supported by GPU.\n", blocksPerGrid, maxBlocks);
        printf("GPUMPI: Exitting...\n");
        exit(1);
    }

    CudaMPI::initError();

    // convert the argv array into memory inside the an UM allocated buffer
    void* argvInUnifiedMemory = copyArgsToUnifiedMemory(argcWithoutGPUMPI,argv);

    // allocate memory for communication
    CudaMPI::SharedState::Context sharedStateContext;
    sharedStateContext.numThreads = numProcs;
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
        threadPrivateStateContext[device].pendingBufferSize = pendingBufferSize;

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

    // increase stack size
    CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, stackSize));
    // increase heap size
    size_t memFree = 0;
    size_t memTotal = 0;
    CUDA_CHECK(cudaMemGetInfo(&memFree, &memTotal));
    printf("GPUMPI: memFree = %zu, memTotal = %zu\n", memFree, memTotal);

    if (heapSize == 0) {
        // leave some free memory for CUDA internal implementation
        // otherwise kernel will not be launched
        double usageRatio = 0.8; 
        printf("GPUMPI: Heap memory requirements are not specified\n");
        printf("GPUMPI: Using %d %% of free memory for the heap\n", (int)(usageRatio * 100));
        heapSize = memFree * usageRatio; 
    }

    printf("GPUMPI: Requested heap memory size is %zu bytes\n", heapSize);
    CUDA_CHECK(cudaDeviceSetLimit(cudaLimitMallocHeapSize, heapSize));

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

    CudaMPI::printLastError();

    // make sure that everything is ok after kernel launch
    CUDA_CHECK(cudaEventQuery(kernelFinishEvent));

    std::cerr << "GPUMPI: Releasing resources" << std::endl;

    // release all resources

    CUDA_CHECK(cudaFree(argvInUnifiedMemory));
    CudaMPI::freeError();

    //MPI_CHECK(MPI_Finalize());

    std::cerr << "GPUMPI: MPI finished!" << std::endl;

    return sharedState->returnValue;
}

