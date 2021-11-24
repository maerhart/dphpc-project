#include <atomic>
#include <assert.h>
#include <iostream>

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

struct superblock {
    std::atomic<int> counter;

};
__device__ void* __gpu_malloc_v1(size_t size) {
    void* ptr = malloc(size);
    #ifndef NDEBUG
    if (!ptr) {
        printf("GPUMPI: malloc failed to allocate %llu bytes on device\n", (long long unsigned)size);
    }
    #endif
    return ptr;
}

__device__ void __gpu_free_v1(void* memptr) {
    free(memptr);
}



// repeatedly allocate individual ints and sum them up
__global__ void test(int *resulting_sums) {


	int* val = (int*)malloc_baseline(sizeof(int));
	*val = 5;
	//printf("test");
	int id = (blockIdx.x*blockDim.x + threadIdx.x);
	resulting_sums[id] = *val;
	free_baseline(val);

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
	for (int i = 0; i < total_threads; ++i) {
		if (resulting_sums[i] != 5) {
			passed = false;
		}
	}

	if (passed) {
		std::cout << "Tests passed" << std::endl;
	} 
	else {
		std::cout << "Failed" << std::endl;
	}

	return 0;
}
