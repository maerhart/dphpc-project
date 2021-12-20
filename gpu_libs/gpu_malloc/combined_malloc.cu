#include "no_header_malloc.cuh"
#include "warp_malloc.cuh"
#include <stdint.h>
#include <stdlib.h>

__device__ void *combined_malloc(size_t size, bool coalesced) {
    if(coalesced) {
        return malloc_v6(size);
    } else {
        return malloc_v5(size);
    } 
}

__device__ void combined_free(void *memptr) {
    if(!free_v6(memptr)) {
        free_v5(memptr);
    }
}
