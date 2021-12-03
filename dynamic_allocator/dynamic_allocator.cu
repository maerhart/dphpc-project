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
	if (count == 0) free(header);
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

/**
 * Malloc for entire warp
 *
 * If we know that 
 *    - each thread frees its own malloced pointers
 *		-> can only let thread 0 free and simplify logic a lot
 *    - when threads free pointers together, these pointers were also allcoated together, not swapped vertically (across time, not threads)
 *              -> can simplify free logic





	https://developer.nvidia.com/blog/cooperative-groups/

	Also, when the tile size matches the hardware warp size, the compiler can elide the synchronization while still ensuring correct memory instruction ordering to avoid race conditions. Intentionally removing synchronizations is an unsafe technique (known as implicit warp synchronous programming) that expert CUDA programmers have often used to achieve higher performance for warp-level cooperative operations. Always explicitly synchronize your thread groups, because implicitly synchronized programs have race conditions.

 */
__device__ void* malloc_v3(size_t size, void*** shared_malloc_sizes_and_ptrs) { // assume there exists a shared preallocated arr initialized to NULL
  // requires sizeof(size_t) <= sizof(void*)
  // all threads execute in lock-step

  // always need first two bits of size field for indicating whether has been freed
  // first bit to indicate free
  // second bit to differentiate -1 and -2 from giant free block
  const size_t free_bit_mask = ((size_t) 1) << (4 * sizeof(size_t) - 1);
  const size_t superblock_bit_mask = ((size_t) 1) << (4 * sizeof(size_t) - 2);
  if ((free_bit_mask | superblock_bit_mask) & size) {
    return NULL;
  }

  int lane_id = threadIdx.x % 32;
  int warp_id = threadIdx.x / 32;

  int thread_id = threadIdx.x; // assumes 1-d indexing


  // let thread 0 allocate for every thread. Before every returned ptr, there is the size of the prev block, or superblock_bit_mask for lane 0's memory piece

  // ptr to shared memory where we can put the address to return from malloc
  void** malloc_addr_ptr = shared_malloc_size_and_ptrs[thread_id];
  // ptr to shared memory where we save how much space this thread needs
  size_t* thread_size_ptr = (size_t*) malloc_addr_ptr;

  *thread_size_ptr = size;

  if (lane_id == 0) { // potentially do this for every thread where thread before does not malloc
  	size_t sizes_total = 0;
  	// find sizes
  	for (int i = 0; i < 32; i++) {
		sizes_total += *((size_t*) shared_malloc_size_and_ptrs[thread_id + i]);
  	}
	sizes_total += 32 * sizeof(size_t); //header for each mini block

	void* curr_ptr = malloc(sizes_total);

	if (curr_ptr == NULL) {
		for (int i = 0; i < 32; i++) {
			*(shared_malloc_size_and_ptrs[thread_id + i]) = NULL;
		}
	} else {
		// write size of prev block into headers and give all threads their addresses
		size_t size_last = -1;
		for (int i = 0; i < 32; i++) {
			*((size_t*) curr_ptr) = size_last;
			void* shared_arr_entry = shared_malloc_size_and_ptr[thread_id +i];
			size_last = *((size_t*) shared_arr_entry);
			*shared_arr_entry = curr_ptr;
			curr_ptr += size_last;
		}
	}
  }

  void* res = *malloc_addr_ptr;

  // reset to 0 to be resuble
  *malloc_addr_ptr = NULL;

  return res;
}



__device__ void free_v3(void* memptr) {
  /*
   * Case 1: some blocks that were allocated together are being freed but not all
   *         -> cannot free superblock
   *	     -> just indicate that blocks are free (by leading 1 in the size field)
   * Case 2: all (remaining) blocks that were allocated together are being freed
   *	     -> need to free superblock
   */

   const size_t free_bit_mask = ((size_t) 1) << (4 * sizeof(size_t) - 1);

   int lane_id = threadIdx.x % 32;
   int warp_id = threadIdx.x / 32;

   int thread_id = threadIdx.x; // assumes 1-d indexing

   const size_t free_bit_mask = ((size_t) 1) << (4 * sizeof(size_t) - 1);
   const size_t superblock_bit_mask = ((size_t) 1) << (4 * sizeof(size_t) - 2);

   size_t* header_ptr = ((size_t*) memptr) - 1;

   // mark own block as free
   *header_ptr = (*header_ptr) | free_bit_mask;

   // each threads walks back to the start of the superblock and counts how many free blocks there are
   // only one thread can find 32 free blocks -> it frees
   // TODO this is incorrect. what if last block frees, then later other blocks are freed IDEA: have bit for is_last_block and pass along
   int free_blocks = 1;
   while (!((*header_ptr) & superblock_bit_mask)) {// while not at superblock
        size_t size_prev_block = (*header_ptr) & ~free_bit_mask; // no need to mask superblock bits as not at superblock
	header_ptr = ((size_t*) (((char*) header_ptr) - size_prev_block)) - 1;
	if (!(*header_ptr) & free_bit_mask) {
	  	break;
	}
	free_blocks++;
   }

   if (free_blocks == 32) {
   	// can free als all blocks part of superblock are free
    	// only one thread can ever reach this
     	free(header_ptr);
   }
}
