#include <iostream>
#include "dynamic_allocator.cu"
#include "warp_malloc.cu"

#define MALLOC malloc_v4
#define FREE free_v4

// allocate one int per thread and set to threadId
__global__ void test(int *resulting_ids) {
    int id = (blockIdx.x*blockDim.x + threadIdx.x);
    int* val = (int*) MALLOC(sizeof(int));
    *val = id;
    resulting_ids[id] = *val;
    FREE(val);
}

int main(int argc, char* argv[]) {
    // run some simple unit tests, only in debug mode!
    //int blocks = 100;
    int blocks = 1;
    int threads_per_block = 32;
    int total_threads = blocks * threads_per_block;

    int resulting_ids[total_threads];
    int *d_resulting_ids;
    cudaMalloc(&d_resulting_ids, total_threads*sizeof(int));

    test<<<blocks, threads_per_block>>>(d_resulting_ids);
    cudaDeviceSynchronize(); // to allow for printf in kernel code

    // check for error
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
       std::cout << "CUDA ERROR: " << cudaGetErrorString(err) << std::endl;
       return 0;
    }

    cudaMemcpy(resulting_ids, d_resulting_ids, total_threads*sizeof(int), cudaMemcpyDeviceToHost);

    // sum up all ids, should match to the sum from 0 to total_threads
    bool passed = true;
    int sum = 0;
    for (int i = 0; i < total_threads; ++i) {
        sum += resulting_ids[i];
    }
    // sum up 0 to total_threads
    passed = sum == (total_threads - 1) * (total_threads) / 2;

    if (passed) {
        std::cout << "Tests passed" << std::endl;
    }
    else {
        std::cout << "Failed" << std::endl;
    }

    return 0;
}
