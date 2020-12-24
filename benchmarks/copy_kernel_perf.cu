#include <cuda.h>

#include <cstdio>
#include <chrono>

#define CUDA_CHECK(expr) \
    do { \
        cudaError_t err = (expr); \
        if (err != cudaSuccess) { \
            printf("CUDA ERROR %s:%d %s %s\n", __FILE__, __LINE__, #expr, cudaGetErrorString(err)); \
            abort(); \
        } \
    } while (0)

using hrclock = std::chrono::high_resolution_clock;

template <typename T>
auto nanoseconds(T x) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(x);
}
 

__global__ void copyKernel( void* __restrict__ dst, const void* __restrict__ src, size_t size) {
    size_t start = threadIdx.x + blockIdx.x * blockDim.x;
    size_t step = blockDim.x * gridDim.x;
    
    uint64_t* d = (uint64_t*) dst;
    uint64_t* s = (uint64_t*) src;
    
    for (size_t i = start; i < size / sizeof(uint64_t); i += step) {
        d[i] = s[i];
    }
}
    
int main() {
    size_t bufferSize = 1ull << 28;
    char* srcBuffer = nullptr;
    char* dstBuffer = nullptr;
    CUDA_CHECK(cudaMalloc(&srcBuffer, bufferSize));
    CUDA_CHECK(cudaMalloc(&dstBuffer, bufferSize));
    
    {
        auto timeStart = hrclock::now();
        CUDA_CHECK(cudaMemcpy(dstBuffer, srcBuffer, bufferSize, cudaMemcpyDefault));
        CUDA_CHECK(cudaDeviceSynchronize());
        auto timeEnd = hrclock::now();
        double elapsedTime = nanoseconds(timeEnd - timeStart).count();
        printf("cudaMemcpy %d MB %lg s %lg GB/s\n", int(bufferSize / (1ull << 20)), elapsedTime / 1e9, bufferSize / elapsedTime);
    }
    
    {
        auto timeStart = hrclock::now();
        CUDA_CHECK(cudaMemcpyAsync(dstBuffer, srcBuffer, bufferSize, cudaMemcpyDefault, 0));
        CUDA_CHECK(cudaDeviceSynchronize());
        auto timeEnd = hrclock::now();
        double elapsedTime = nanoseconds(timeEnd - timeStart).count();
        printf("cudaMemcpyAsync %d MB %lg s %lg GB/s\n", int(bufferSize / (1ull << 20)), elapsedTime / 1e9, bufferSize / elapsedTime);
    }
    
    for (int gridSize = 1; gridSize <= 4096; gridSize *= 2) {
        for (int blockSize = 32; blockSize <= 1024; blockSize *= 2) {
            auto timeStart = hrclock::now();
            copyKernel<<<gridSize, blockSize>>>(dstBuffer, srcBuffer, bufferSize);
            CUDA_CHECK(cudaPeekAtLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            auto timeEnd = hrclock::now();
            double elapsedTime = nanoseconds(timeEnd - timeStart).count();
            printf("copyKernel<<<%d,%d>>> %d MB %lg s %lg GB/s\n", gridSize, blockSize, int(bufferSize / (1ull << 20)), elapsedTime / 1e9, bufferSize / elapsedTime);
        }
    }
    
    CUDA_CHECK(cudaFree(srcBuffer));
    CUDA_CHECK(cudaFree(dstBuffer));
}
