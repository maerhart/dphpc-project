#include <assert.h>
#include <iostream>
#include "cooperative_groups.h"

// baseline using std malloc/free
__device__ void* malloc_baseline(size_t size) {
    void* ptr = malloc(size);
    #ifndef NDEBUG
    if (!ptr) {
        printf("GPUMPI: malloc failed to allocate %llu bytes on device\n", (long long unsigned)size);
    }
    #endif
    return ptr;
}

__device__ void free_baseline(void *memptr) {
    free(memptr);
}

// v1: allocate same sizes for future blocks

struct s_header {
    int counter;
};


// memory layout of a superblock
// s_header, blocksize x [pointer to s_header, data]
__device__ void* malloc_v1(size_t size) {
	__shared__ void* superblock;

	if (threadIdx.x == 0) {
		// allocate new superblock
		superblock = malloc(sizeof(s_header) + blockDim.x * (sizeof(s_header*) + size));
		if (!superblock) return NULL;
		
		// initialize header	
		struct s_header* header;
		header = (s_header*)superblock;
		header->counter = blockDim.x;
	}
	__syncthreads();

	// ptr to individual memory offset
    s_header* ptr = (s_header*)((char*)superblock + sizeof(s_header) + threadIdx.x * (size + sizeof(s_header*)));
	// set pointer to superblock header
	*ptr = *(s_header*)superblock;
	// return the pointer to the data section
	return (void*)(ptr + 1);
}

__device__ void free_v1(void* memptr) {
	// decrease counter
	s_header* header = (s_header*)memptr - 1;    
	int count = atomicSub(&(header->counter), 1);
	
	// last thread frees superblock
	if (count == 0) free(header);
}



// allocate one int per thread and set to threadId
__global__ void test(int *resulting_ids) {
	
	int id = (blockIdx.x*blockDim.x + threadIdx.x);
	int* val = (int*)malloc_v1(sizeof(int));
	*val = id;
	resulting_ids[id] = *val;
}

int main(int argc, char* argv[]) {
	// run some simple unit tests, only in debug mode!
	int blocks = 100;
	int threads_per_block = 32;
	int total_threads = blocks * threads_per_block;

	int resulting_ids[total_threads];
	int *d_resulting_ids;
	cudaMalloc(&d_resulting_ids, total_threads*sizeof(int));	

	test<<<blocks, threads_per_block>>>(d_resulting_ids);
	cudaDeviceSynchronize(); // to allow for printf in kernel code
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
