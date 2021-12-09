#include "dyn_malloc.cuh"
#include "cooperative_groups.h"
#include <stdint.h>

// Let's one thread allocate all memory for each block if
// coalesced is set to true.

// TODO:
// * blockDim does not represent actual threads in the block.
// * ThreadIdx is per block, right?
// * Check synchronization/data races
// * How to deal with mem? should it be __shared__ ?

__shared__ static void *mem;


__device__ void *dyn_malloc(size_t size, bool coalesced) {
    if(!coalesced) {
        // Ensure compatibility with coalesced case
        // | counter 4B | ptr to counter 8B | returned ptr... |
        void *ptr = malloc(size +256);//+ sizeof(int*) + sizeof(int));
        int32_t *counter = (int32_t*) ptr;
        int32_t **counter_ptr = (int32_t**)(counter+32);
        *counter = 1;
        *counter_ptr = counter;
        return counter_ptr+16;
    }

    size_t allocatePerThread = size % 128;
    allocatePerThread = (128 - allocatePerThread) % 128 + 128 + size;
    if (threadIdx.x == 0) {
        mem = malloc(allocatePerThread * blockDim.x + 128);
        // Initialize counter
        *(int32_t*)mem = blockDim.x;
    }

    auto block = cooperative_groups::this_thread_block();

    block.sync();

    void *ptr = (char*)mem + 128 + 128 + threadIdx.x * allocatePerThread;
    int32_t **counter_ptr = (int32_t**) ((char*)ptr-128);
    *counter_ptr = (int32_t*)mem;

    block.sync();

    return ptr;
}

__device__ void dyn_free(void *memptr) {
    int32_t *counter_ptr = *(((int32_t**)memptr)-16);
    int32_t counter = atomicAdd(counter_ptr, -1);

    if (counter==1) {
        free(counter_ptr);
    }
}
