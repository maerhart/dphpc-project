#include "cuda_runtime.h"
#include <device_launch_parameters.h>
#include <stdio.h>

__global__ void sum_values_and_allocate(long value, int* res) {
    int val = 0;
    
    int *array = (int *)malloc(value * sizeof(int));
    int id = blockIdx.x * blockDim.x + threadIdx.x;
   
    if(array == NULL) {
	val = 1;
    }
    for(long i = 0; i < value; i++) {
	if(array != NULL) array[i] = i/3;
    }
    free(array);
    res[id] = val;
}

__global__ void sum_values_in_allocated_array(int* array, long range, int* res) {
    int val = 0;

    long start = blockIdx.x * blockDim.x + threadIdx.x;

    if(array == NULL) {
	val = 1;
    }
    for(long i = start; i < start + range; i++) {
	if(array != NULL) array[i] = i/3;
    }
    res[start] = val;
}

int main(int argc, char **argv) {
    cudaError_t cuda_status;
    int coalesced = (atoi(argv[2]) == 1) ? 1 : 0;
    char *simple_output = argv[5];
    if(!simple_output) {
	    printf("%s Starting...\n", argv[0]);
    }
    long ints = 1L << atoi(argv[1]);
    if(!simple_output) {
	if(coalesced > 0) {
	    printf("Benchmarking Coalesced for %ld ints\n", ints);
	} else {
	    printf("Benchmarking non coalesced for %ld ints\n", ints);
	}
    }

    int threads_per_block = 1 << atoi(argv[3]);
    // max value is 1024
    threads_per_block = (threads_per_block > 1024) ? 1024 : threads_per_block;
    int blocks = 1 << atoi(argv[4]);
    // max value of concurrently executable blocks is 192 (largest power of 2 is 128)
    blocks = (blocks > 65535) ? 65535 : blocks;

    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    if(!simple_output) {
        printf("Using Device %d: %s\n", dev, deviceProp.name);
    }
    cudaSetDevice(dev);

    //2^20
    int total_threads = threads_per_block * blocks;

    int* res;
    cudaMalloc((void**)&res, total_threads * sizeof(int));
    cudaMemset(res, 0, total_threads * sizeof(int));
    int* resCPU = (int *) malloc(total_threads * sizeof(int));
    long allocation_size = ints;
    long allocation_per_thread = allocation_size / total_threads;
    allocation_per_thread = (allocation_per_thread < 1) ? 1 : allocation_per_thread;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaDeviceSynchronize();
    float ms = 0;
    if(!coalesced) {
        cudaEventRecord(start);
        sum_values_and_allocate<<<blocks, threads_per_block>>>(allocation_per_thread, res);
        cudaEventRecord(stop);

        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);
        cuda_status = cudaDeviceSynchronize();
        if(!simple_output) {
            printf("cudaMalloc(%ld ints/%ld bytes) over %d threads (%d blocks, %d threads per block): Time elapsed %f ms\n", allocation_per_thread, allocation_per_thread * sizeof(int), total_threads, blocks, threads_per_block, ms);
	} else {
	    printf("%d %d %f ", blocks, threads_per_block, ms);
	}
    } else {
        cudaEventRecord(start);
        int* array;
        cudaMalloc((void**)&array, allocation_size*sizeof(int));
        sum_values_in_allocated_array<<<blocks, threads_per_block>>>(array, allocation_per_thread, res);
        cudaFree(array);
        cudaEventRecord(stop);

        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);
        cuda_status = cudaDeviceSynchronize();
        if(!simple_output) {
            printf("cudaMalloc(%ld ints/%ld bytes) coalesced (%d blocks, %d threads per block): Time elapsed %f ms\n", allocation_size, allocation_size * sizeof(int), blocks, threads_per_block, ms);
	} else {
	    printf("%d %d %f ", blocks, threads_per_block, ms);
	}
    }
    cudaMemcpy(resCPU, res, total_threads * sizeof(int), cudaMemcpyDeviceToHost);
    int malloc_failures = 0;
    for(int i = 0; i < total_threads; i++) malloc_failures += resCPU[i];

    if(!simple_output) {
	printf("%d threads failed to allocate memory\n", malloc_failures);
    } else {
	printf("%d\n", malloc_failures);
    }

    if (cuda_status != cudaSuccess) {
        printf("Error: %d %s\n", cuda_status, cudaGetErrorString(cuda_status));
        exit(1);
    }
    cudaFree(res);
    return 0;
}
