#include "dyn_malloc.cuh"
#include "cooperative_groups.h"
#include <stdint.h>
#include <stdlib.h>

// Let's one thread allocate all memory for each block if
// coalesced is set to true.

// TODO:
// * blockDim does not represent actual threads in the block.
// * ThreadIdx is per block, right?
// * Check synchronization/data races
// * How to deal with mem? should it be __shared__ ?

__shared__ static void *mem;  // max 32 warps

#define ALIGN 0
#define PAD 4416

__device__ void *round_to_alignment(void *addr) {
    long offset = (long)addr;
    if (ALIGN != 0) {
        offset %= ALIGN;
    } else {
        offset = 0;
    }
    return (void*)(ALIGN - offset + (long)addr);
}

//__device__ void *dyn_malloc(size_t size, bool coalesced) {
//    int warpno = threadIdx.x / 32;
//    if (threadIdx.x % 32 == 0) {
//        mem[warpno] = malloc(size * 32 + ALIGN);
//        //printf("%d %ld\n", warpno, (long)mem[warpno]);
//    } else {
//        //printf("");
//    }
//
//    __syncwarp();
//
//    void *ptr = (char*)round_to_alignment(mem[warpno]) + (threadIdx.x % 32) * size;
//    return ptr;
//}

//__device__ int mapWarpNo(int wn) {
//    switch(wn) {
//    case 0: return 0;
//    case 1: return 13;
//    case 2: return 1;
//    case 3: return 8;
//    case 4: return 14;
//    case 5: return 5;
//    case 6: return 29;
//    case 7: return 17;
//    case 8: return 12;
//    case 9: return 15;
//    case 10: return 26;
//    case 11: return 28;
//    case 12: return 19;
//    case 13: return 23;
//    case 14: return 3;
//    case 15: return 18;
//    case 16: return 16;
//    case 17: return 6;
//    case 18: return 7;
//    case 19: return 2;
//    case 20: return 10;
//    case 21: return 4;
//    case 22: return 9;
//    case 23: return 11;
//    case 24: return 27;
//    case 25: return 22;
//    case 26: return 25;
//    case 27: return 24;
//    case 28: return 20;
//    case 29: return 30;
//    case 30: return 21;
//    case 31: return 31;
//    default: return 0;
//    }
//}

//__shared__ static void *mp[32];  // max 32 warps
//
//__device__ void *dyn_malloc(size_t size, bool coalesced) {
//    int warpno = threadIdx.x / 32;
//    size_t allocatePerWarp = size * 32;
//    if (threadIdx.x % 32 == 0) {
//        mp[warpno] = malloc(size*32);
//    }
//
//    auto block = cooperative_groups::this_thread_block();
//    block.sync();
//
//    void *ptr = (char*)mp[warpno] + (threadIdx.x%32)*4; //(threadIdx.x / 32) * allocatePerWarp + (threadIdx.x % 32) * size;
//
//    //block.sync();
//    return ptr;
//}

__device__ void *dyn_malloc(size_t size, bool coalesced) {
    int warpno = threadIdx.x / 32;
    size_t allocatePerWarp = size * 32;// + 4416 - (size * 32) % 4416;
    if (threadIdx.x == 0) {
        mem = malloc(allocatePerWarp * blockDim.x/32 + ALIGN);
        //for (int i = 0; i < 32;++i) {
        //    mem[i] = malloc(size*32);
        //    if (i>0) printf("%ld\n", abs((long)mem[i]-(long)mem[i-1]));
        //}
    }

    auto block = cooperative_groups::this_thread_block();
    block.sync();

    void *ptr = (char*)mem + warpno*allocatePerWarp + (threadIdx.x%32)*4; //(threadIdx.x / 32) * allocatePerWarp + (threadIdx.x % 32) * size;

    //block.sync();
    return ptr;
}

__device__ void dyn_free(void *memptr) {
    //auto block = cooperative_groups::this_thread_block();
    //block.sync();
    //if (threadIdx.x == 0) {
    //    free((char*)memptr);
    //}
}

//__device__ void *dyn_malloc(size_t size, bool coalesced) {
//
//    if(!coalesced) {
//        // Ensure compatibility with coalesced case
//        // | counter 4B | ptr to counter 8B | returned ptr... |
//        void *ptr = malloc(size +256);//+ sizeof(int*) + sizeof(int));
//        int32_t *counter = (int32_t*) ptr;
//        int32_t **counter_ptr = (int32_t**)(counter+32);
//        *counter = 1;
//        *counter_ptr = counter;
//        return counter_ptr+16;
//    }
//
//    size_t allocatePerThread = size % 128;
//    allocatePerThread = (128 - allocatePerThread) % 128 + 128 + size;
//    if (threadIdx.x == 0) {
//        mem = malloc(allocatePerThread * blockDim.x + 128);
//        // Initialize counter
//        *(int32_t*)mem = blockDim.x;
//    }
//
//    auto block = cooperative_groups::this_thread_block();
//
//    block.sync();
//
//    void *ptr = (char*)mem + 128 + 128 + threadIdx.x * allocatePerThread;
//    int32_t **counter_ptr = (int32_t**) ((char*)ptr-128);
//    *counter_ptr = (int32_t*)mem;
//
//    block.sync();
//
//    return ptr;
//}
//
//__device__ void dyn_free(void *memptr) {
//    int32_t *counter_ptr = *(((int32_t**)memptr)-16);
//    int32_t counter = atomicAdd(counter_ptr, -1);
//
//    if (counter==1) {
//        free(counter_ptr);
//    }
//}
