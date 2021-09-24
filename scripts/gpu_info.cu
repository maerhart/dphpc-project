#include <cuda.h>
#include <cstdio>
#include <cooperative_groups.h>

#define CUDA_CHECK(expr) do {\
    cudaError_t err = (expr);\
    if (err != cudaSuccess) {\
        printf("CUDA ERROR: %s:%d : %s <%s>  %s\n", __FILE__, __LINE__, #expr, cudaGetErrorName(err), cudaGetErrorString(err)); \
        exit(1); \
    }\
} while(0)

__global__ void dummy_kernel() {
    printf("begin\n");
    cooperative_groups::this_grid().sync();
    printf("end\n");
}

int main() {
    int computeCapabilityMajor = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&computeCapabilityMajor, cudaDevAttrComputeCapabilityMajor, /* device */ 0));
    int computeCapabilityMinor = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&computeCapabilityMinor, cudaDevAttrComputeCapabilityMinor, /* device */ 0));

    bool hasIndependentThreadScheduling = computeCapabilityMajor >= 7;
    int blockSizeLimit = 0; // unlimited
    if (!hasIndependentThreadScheduling) {
        blockSizeLimit = 1;
    }

    int minGridSize = 0;
    int blockSize = 0;
    CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, dummy_kernel, /*dynamicSMemSize*/ 0, /*blockSizeLimit*/ blockSizeLimit));

    int blocksPerMP = -1;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerMP, dummy_kernel, blockSize, /*sharedMem*/ 0));

    int multiProcessorCount = -1;
    CUDA_CHECK(cudaDeviceGetAttribute(&multiProcessorCount, cudaDevAttrMultiProcessorCount, /*device*/ 0));

    printf("maxCooperativeThreads %d\n", multiProcessorCount * blocksPerMP * blockSize);
    printf("computeCapabilityMajor %d\n", computeCapabilityMajor);
    printf("computeCapabilityMinor %d\n", computeCapabilityMinor);

    return 0;
}