#include <atomic>
#include <assert.h>

// baseline using std malloc/free
__device__ void* __gpu_malloc_baseline(size_t size) {
    void* ptr = malloc(size);
    #ifndef NDEBUG
    if (!ptr) {
        printf("GPUMPI: malloc failed to allocate %llu bytes on device\n", (long long unsigned)size);
    }
    #endif
    return ptr;
}

__device__ void* __gpu_free_baseline(void *memptr) {
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

__device__ void* __gpu_free_v1(void* memptr) {
    free(memptr);
}



// repeatedly allocate individual ints and sum them up
void test() {
	const int num_ints = 100; // need enough space for num_ints ints and free list and headers

	// repeat test multiple times
	for (int num_runs = 0; num_runs < 10; ++num_runs) {

		// array with results
		int* result[num_ints];
		int* reference[num_ints];


		// fill available space with ints 0, 1, ...
		for (int i = 0; i < num_ints; ++i) {
			// custom malloc
			int* val = (int*)__gpu_malloc_baseline(sizeof(int));
			assert(val); // No null pointers
			*val = i;
			result[i] = val;

			// reference
			int* val_ref = (int*)malloc(sizeof(int));
			assert(val_ref); // No null pointers
			*val_ref = i;
			reference[i] = val_ref;
		}

		// compare results
		int res_sum = 0;
		int ref_sum = 0;
		for (int i = 0; i < num_ints; ++i) {

			res_sum += *(result[i]);
			ref_sum += *(reference[i]);

			// correct value stored
			assert(*(result[i]) == i);
			assert(*(reference[i]) == i);
		}

		// correct sum
		assert(ref_sum == (num_ints - 1) * (num_ints) / 2); // analytic solution
		assert(res_sum == ref_sum); // reference solution

		// free for the next round
		for (int i = 0; i < num_ints; ++i) {
			__gpu_free_baseline(result[i]);
			free(reference[i]);
		}

	}
}

int main(int argc, char* argv[]) {
	// run some simple unit tests, only in debug mode!
	test();

	std::cout << "Tests passed" << std::endl;

	return 0;
}