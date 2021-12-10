#include "cuda_runtime.h"
#include <device_launch_parameters.h>
#include <stdio.h>

__global__ void sum_values_in_allocated_array(int* array, int range, int* res) {
    int sum = 0;
    int start = blockIdx.x * blockDim.x + threadIdx.x;
    for(int i = start; i < start + range; i++) {
        sum += (i+array[i]);
    }
    res[start] = sum;
}

__global__ void sum_values_and_allocate(int size, int* res) {
    int sum = 0;
    int *array;
    
    cudaMalloc(&array, size);
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    for(int i = 0; i < size; i++) {
        sum += (i+array[i]);
    }
    cudaFree(array);
    res[id] = sum;
}

int main(int argc, char **argv) {
    cudaError_t cuda_status;
    printf("%s Starting...\n", argv[0]);
    int coalesced = (atoi(argv[2]) == 1) ? 1 : 0;
    int bytes = 1 << atoi(argv[1]);
    if(coalesced > 0) {
        printf("Benchmarking Coalesced for %d bytes\n", bytes);
    } else {
        printf("Benchmarking non coalesced for %d bytes\n", bytes);
    }
    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    cudaSetDevice(dev);

    int threads_per_block = 1024;
    int blocks = 1024;
    int total_threads = threads_per_block * blocks;

    int* res;
    cudaMalloc((void**)&res, total_threads * sizeof(int));
    int allocation_size = bytes;
    int allocation_per_thread = allocation_size / total_threads;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaDeviceSynchronize();
    float ms = 0;
    if(coalesced) {
        cudaEventRecord(start);
        sum_values_and_allocate<<<blocks, threads_per_block>>>(allocation_per_thread, res);
        cudaEventRecord(stop);

        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);
        cuda_status = cudaDeviceSynchronize();
        printf("cudaMalloc(%d) over %d threads: Time elapsed %f ms\n", allocation_per_thread, total_threads, ms);
    } else {
        cudaEventCreate(&start);
        int* array;
        cudaMalloc((void**)&array, allocation_size);
        sum_values_in_allocated_array<<<blocks, threads_per_block>>>(array, allocation_per_thread / sizeof(int), res);
        cudaFree(array);
        cudaEventRecord(stop);

        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);
        cuda_status = cudaDeviceSynchronize();
        printf("cudaMalloc(%d) coalesced: Time elapsed %f ms\n", allocation_size, ms);
    }

    if (cuda_status != cudaSuccess) {
        printf("Error: %d\n", cuda_status);
        exit(1);
    }
    cudaFree(res);
    return 0;
}
