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



__device__ int num_avail_blocks = 0;
__device__ void* malloc_v1(size_t size) {
	
	__shared__ void* superblock;

	if (num_avail_blocks < 1) {
		// allocate new superblock
		int n = 32;
		superblock = malloc(sizeof(s_header) + n * (sizeof(s_header*) + size));
		
		struct s_header* header;
		header = (s_header*)superblock;
		//s_header *header = (s_header*)superblock;
		header->counter = blockDim.x;
		
		//s_header** ref;
		//ref = (s_header**)((s_header*)superblock + 1);
		//*ref = header;
		

			
		//*((s_header*)superblock + 1) = *header;
		//void* ptr = (char*)superblock + sizeof(s_header) + sizeof(s_header*);
		
		//return ptr; 
	}
	__syncthreads();
	//auto block = cooperative_groups::this_thread_block();
	//block.sync();



    void* ptr = (char*)superblock + sizeof(s_header) + threadIdx.x * (size + sizeof(s_header*)) + sizeof(s_header*);
	s_header** header = (s_header**)((char*)ptr - sizeof(s_header*));
	*header = (s_header*)superblock;

	return ptr;
}

__device__ void free_v1(void* memptr) {
	s_header* header = ((s_header*)memptr - 1);    
	int count = atomicAdd(&(header->counter), -1);

	if (count == 0) free(header);
}



// repeatedly allocate individual ints and sum them up
__global__ void test(int *resulting_sums) {
	
	int id = (blockIdx.x*blockDim.x + threadIdx.x);

	int* val = (int*)malloc_v1(sizeof(int));
	*val = id;
	//printf("test");
	resulting_sums[id] = *val;
	free_v1(val);

	/*
	// array with results
	const int num_ints = 100;
	int* result[num_ints];

	// fill available space with ints 0, 1, ...
	for (int i = 0; i < num_ints; ++i) {
		// custom malloc
		int* val = (int*)malloc_baseline(sizeof(int));
		assert(val); // No null pointers
		*val = i;
		result[i] = val;

	}

	// compare results
	int res_sum = 0;
	for (int i = 0; i < num_ints; ++i) {

		res_sum += *(result[i]);

		// correct value stored
		assert(*(result[i]) == i);
	}

	// correct sum
	assert(res_sum == (num_ints - 1) * (num_ints) / 2); // analytic solution
	int id = (blockIdx.x*blockDim.x + threadIdx.x);
	resulting_sums[id] = res_sum;

	// free for the next round
	for (int i = 0; i < num_ints; ++i) {
		free_baseline(result[i]);
	}
	*/

}

int main(int argc, char* argv[]) {
	// run some simple unit tests, only in debug mode!
	int blocks = 1;
	int threads_per_block = 32;
	int total_threads = blocks * threads_per_block;

	int resulting_sums[total_threads];
	int *d_resulting_sums;
	cudaMalloc(&d_resulting_sums, total_threads*sizeof(int));	

	test<<<blocks, threads_per_block>>>(d_resulting_sums);
	cudaDeviceSynchronize(); // to allow for printf in kernel code
	cudaMemcpy(resulting_sums, d_resulting_sums, total_threads*sizeof(int), cudaMemcpyDeviceToHost);

	bool passed = true;
	int sum = 0;
	for (int i = 0; i < total_threads; ++i) {
		sum += resulting_sums[i];
		if (resulting_sums[i] != 5) {
			//passed = false;
		}
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
