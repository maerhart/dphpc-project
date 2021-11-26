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
		int size_superblock = sizeof(s_header) + blockDim.x * (sizeof(s_header*) + size);
		superblock = malloc(size_superblock);
		if (superblock == NULL) {
			printf("V1: failed to allocate %llu bytes on device\n", (long long unsigned)(size_superblock));
			return NULL;
		}
		// initialize header	
		struct s_header* header;
		header = (s_header*)superblock;
		header->counter = blockDim.x;
	}
	__syncthreads();

	if (superblock == NULL) return NULL;

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
