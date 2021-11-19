#include "dyn_malloc.cuh"
#include "cooperative_groups.h"

// Let's one thread allocate all memory for each block if
// coalesced is set to true.

// TODO:
// * blockDim does not represent actual threads in the block.
// * ThreadIdx is per block, right?
// * Check synchronization/data races
// * How to deal with mem? should it be __shared__ ?

__shared__ static void **mem;


__device__ void *dyn_malloc(size_t size, bool coalesced) {
    if(!coalesced) {
        // Ensure compatibility with coalesced case
        // | counter 4B | ptr to counter 8B | returned ptr... |
        void *ptr = malloc(size + sizeof(int*) + sizeof(int));
        int *counter = (int*) ptr;
        int **counter_ptr = (int**)(counter+1);
        *counter = 1;
        *counter_ptr = counter;
        return counter_ptr+1;
    }

    if (threadIdx.x == 0) {
        *mem = malloc((size + sizeof(int*)) * blockDim.x + sizeof(int));
        // Initialize counter
        **(int**)mem = blockDim.x;
    }

    auto block = cooperative_groups::this_thread_block();

    block.sync();

    void *ptr = (char*)*mem + sizeof(int) + sizeof(int*) + threadIdx.x * (size + sizeof(int*));
    int **counter_ptr = (int**) ((char*)ptr-sizeof(int*));
    *counter_ptr = (int*)*mem;

    block.sync();

    return ptr;
}

__device__ void dyn_free(void *memptr) {
    int *counter_ptr = *((int**)memptr-1);
    int counter = atomicAdd(counter_ptr, -1);

    if (counter==0) {
        free(counter_ptr);
    }
}
