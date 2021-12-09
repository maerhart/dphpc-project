#include <assert.h>
#include <iostream>
#include <stdint.h>
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
	if (count == 1) free(header);
}



// v2: no spaceing between data, use hashmap to know mapping from address to superblock
#define SIZE_HASH_MAP 1000
__shared__ s_header* hashmap[SIZE_HASH_MAP];


// source: https://stackoverflow.com/questions/664014/what-integer-hash-function-are-good-that-accepts-an-integer-hash-key
__device__ uint64_t hash(uint64_t x) {
    x = (x ^ (x >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)) * UINT64_C(0x94d049bb133111eb);
    x = x ^ (x >> 31);
    return x;
}

__device__ void* malloc_v2(size_t size) {
    __shared__ void* superblock;
    if (threadIdx.x == 0) {
        // allocate new superblock
        int size_superblock = sizeof(s_header) + blockDim.x * size;
        superblock = malloc(size_superblock);
        if (superblock == NULL) {
            printf("V2: failed to allocate %llu bytes on device\n", (long long unsigned)(size_superblock));
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
    s_header* ptr = (s_header*)((char*)superblock + sizeof(s_header) + threadIdx.x * size);
    // insert into hashmap
	int h = hash((uintptr_t)ptr) % SIZE_HASH_MAP;
	// todo: proper collision handling
	if (hashmap[h] != 0) {
		printf("V2: hash collision on %i\n", h);
	}
	hashmap[h] = (s_header*)superblock;
	// return the pointer to the data section
    return (void*)ptr;
}

__device__ void free_v2(void* memptr) {
    // get ptr and delete hashmap entry
	int h = hash((uintptr_t)memptr) % SIZE_HASH_MAP;
    s_header* header = hashmap[h];
	hashmap[h] = 0;
	// decrease counter
    int count = atomicSub(&(header->counter), 1);

    // last thread frees superblock
    if (count == 0) free(header);
}


