// nvcc thread_divergence.cu -arch=sm_75 && /opt/cuda/nsight_compute/ncu --metrics smsp__thread_inst_executed_per_inst_executed,smsp__thread_inst_executed,smsp__inst_executed ./a.out

#include <cstdio>
#include <cuda.h>

#define CUDA_CHECK(expr) \
    do { \
        cudaError_t err = (expr); \
        if (err != cudaSuccess) { \
            printf("CUDA_ERROR %s:%d %s %s\n", __FILE__, __LINE__, #expr, cudaGetErrorString(err)); \
            abort(); \
        } \
    } while (0)

__device__ double workload(double value, double factor, int repeats) {
    for (int i = 0; i < repeats; i++) {
        value = value * factor;
        value = value / factor;
    }
    return value;
}

/* 
    This kernel should check thread divergence for a single warp.
    X = workload per thread for workload call
    (32 * X + 10 * X + 15 * X) / (32 * X + 32 * X + 32 * X) = 0.59
    Expected ratio of useful work (thread-non-divergence) 59%
*/
__global__ void kernel1(double* outputs) {
    int rank = threadIdx.x;
    double value = 1.23;
    constexpr int repeats = 10000;
    value = workload(value, 1.1, repeats);
    if (rank < 10) {
        value = workload(value, 1.2, repeats);
    } else if (rank < 25) {
        value = workload(value, 1.3, repeats);
    }

    // prevent optimizing everythig away
    outputs[rank] = value;
}

/* 
    This kernel should check thread divergence for two warps and give average among them.
    X = workload per thread for workload call
    useful work show how much work we use out of possible work done without part of threads sleeping
    first warp: (32X + 4X + 4X + 4X + 4X + 16X) / (32X + 32X + 32X + 32X + 32X + 32X) = (32 * 2X) / (32 * 6X) = 1/3
    second warp: (32X + 32X) / (32X + 32X) = (32 * 2X) / (32 * 2X) = 1
    both warps = (32 * 2X + 32 * 2X) / (32 * 6X + 32 * 2X) = 1/2 
*/
__global__ void kernel2(double* outputs) {
    int rank = blockIdx.x * blockDim.x + threadIdx.x;
    double value = 1.23;
    constexpr int repeats = 10000;
    value = workload(value, 1.1, repeats);
    if (rank < 4) {
        value = workload(value, 1.2, repeats);
    } else if (rank < 8) {
        value = workload(value, 1.3, repeats);
    } else if (rank < 12) {
        value = workload(value, 1.4, repeats);
    } else if (rank < 16) {
        value = workload(value, 1.5, repeats);
    } else {
        value = workload(value, 1.6, repeats);
    }

    // prevent optimizing everythig away
    outputs[rank] = value;
}

/*
    This kernel is supposed to test data-dependency thread-divergence.
    Expected useful work:
    first warp: (32 * 2X) / (32 * 6X) = 1/3
    second warp: (32X + X + 31X) / (32X + 32X + 32X) = 2/3
    overall: (2 + 2) / (6 + 3) = 0.44

*/
__global__ void kernel3(double* condition) {
    int rank = threadIdx.x;
    double cond = condition[rank];
    double value = 1.23;
    constexpr int repeats = 10000;
    value = workload(value, 1.1, repeats);
    if (cond < 1) {
        value = workload(value, 1.2, repeats);
    } else if (cond < 2) {
        value = workload(value, 1.3, repeats);
    } else if (cond < 3) {
        value = workload(value, 1.4, repeats);
    } else if (cond < 4) {
        value = workload(value, 1.5, repeats);
    } else {
        value = workload(value, 1.6, repeats);
    }

    // prevent optimizing everythig away
    condition[rank] = value;
}

/*
    This kernel is expected to measure thread divergence 
    for unequal chunks of load among the threads
    (5 * 2X + 10 * 3X) / (32 * 2X + 32 * 3X) = 0.25
*/
__global__ void kernel4(double* outputs) {
    int rank = threadIdx.x;
    double value = 1.23;
    constexpr int repeats = 10000;
    if (rank < 5) {
        value = workload(value, 1.1, 2 * repeats);
    } else if (rank < 15) {
        value = workload(value, 1.2, 3 * repeats);
    }

    // prevent optimizing everythig away
    outputs[rank] = value;
}

int main() {
    size_t dataSize = 64;
    double* data = nullptr;
    CUDA_CHECK(cudaMallocManaged(&data, dataSize));

    kernel1<<<1,32>>>(data);
    CUDA_CHECK(cudaPeekAtLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    kernel2<<<1,64>>>(data);
    CUDA_CHECK(cudaPeekAtLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    kernel2<<<2,32>>>(data);
    CUDA_CHECK(cudaPeekAtLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    double datacopy[64] = {
        0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3, // 16
        4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4, // 16
        0,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4, // 16
        4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4, // 16
    };
    memcpy(data, datacopy, sizeof(double) * dataSize);

    kernel3<<<1,64>>>(data);
    CUDA_CHECK(cudaPeekAtLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    kernel4<<<1,32>>>(data);
    CUDA_CHECK(cudaPeekAtLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}
