#include <assert.h>
#include <iostream>

__device__ const int WARP_SIZE = 32;

__forceinline__ __device__ unsigned lane_id_asm()
{
    unsigned ret;
    asm volatile ("mov.u32 %0, %laneid;" : "=r"(ret));
    return ret;
}

__forceinline__ __device__ unsigned lane_id() // TODO lane_id not found
{
    unsigned id = threadIdx.x % WARP_SIZE;
    assert(id == lane_id_asm());
    return id;
}

__device__ uint32_t lanemask_lt() { // TODO __lanemask_lt() not found
    return ((uint32_t)1 << lane_id()) - 1;
}

__device__ int active_lane_id(uint32_t active_mask) {
    return __popc(active_mask & lanemask_lt());
}

__device__ bool is_active(int lid, uint32_t active_mask) {
    return (((uint32_t) 1) << lid) & active_mask;
}

/**
 * Safe warp level malloc
 *
 * Precondition: sizeof(size_t) == sizeof(void*) (given on our GPU)
 *
 *  Each block has a header of size sizeof(void*)
 *  with bits for is_superblock, is_free, and is_last_block
 *
 *  TODO check if threadsafe if concurrent frees in different warps/threadblocks of blocks malloced together
 *
 * TODO check current assumption:
 *     - lane_id cannot change
 *     - threads cannot move to different warps
 */
__device__ void* malloc_v4(size_t size) {
    // check preconditions. If this is not given need rewrite bit manipulations
    assert(sizeof(size_t) == sizeof(void*));
    size_t header_size = sizeof(void*);

    const size_t free_bit_mask = ((size_t) 1) << (header_size - 1);
    const size_t superblock_bit_mask = ((size_t) 1) << (header_size - 2);
    const size_t lastblock_bit_mask = ((size_t) 1) << (header_size - 3);

    // adjust size for alignment purposes
    // TODO handle better
    if (size % 8 != 0) {
        size += 8 - size % 8;
    }

    // assert special bits not used
    if ((free_bit_mask | superblock_bit_mask | lastblock_bit_mask) & size) {
        return NULL;
    }

    int my_lane_id = lane_id();

    // retrieve mask of all threads in this warp that are currently executing
    // this instruction. they will perform a coalesced malloc
    uint32_t active_mask = __activemask();
    // count number of 1s
    int n_threads = __popc(active_mask);
    // Find the lowest-numbered active lane
    int elected_lane = __ffs(active_mask) - 1;
    // get id/idx among active lanes
    int my_active_lane_id = active_lane_id(active_mask);

    // find out how much memory each thread needs
    size_t required_size_above = size; // how much all participating threads with lane_id >= own need
    // after step i, required_size_above holds the required size of next i threads
    // (including non-active threads for which the shuffle instruction returns 0)
    for (int i = 1; i < WARP_SIZE - 1; i++) {
        size_t size_i_above = __shfl_down_sync(active_mask, size, i);
        // check if result valid. if not both threads are active and participating
        // in shuffle, then result is undefined
        if ((i + my_lane_id < WARP_SIZE - 1) && is_active(my_lane_id + i, active_mask)) {
            required_size_above += size_i_above;
        }
    }

    __syncwarp(active_mask);

    // the elected_lane holds the total sum of required sizes
    size_t required_size_total = __shfl_sync(active_mask, required_size_above, elected_lane);

    char* malloced_ptr = NULL;

    // perform coalesced malloc
    if (my_lane_id == elected_lane) {
        malloced_ptr = (char*) malloc(required_size_total + n_threads * header_size);
    }

    // broadcast alloced ptr to all lanes
    assert(sizeof(size_t) == sizeof(char*)); // make sure we don't change due to cast
    // need to cast as pointers can't be shuffled
    malloced_ptr = (char*) __shfl_sync(active_mask, (size_t) malloced_ptr, elected_lane);

    // header space required for the threads with lower ids
    int header_size_below = my_active_lane_id * header_size;
    // compute this thread's memory region
    size_t* header_ptr = (size_t*) (malloced_ptr + required_size_total - required_size_above + header_size_below);

    // write header
    if (my_lane_id == elected_lane) {
        // write superblock header
        *header_ptr = superblock_bit_mask;
    } else {
        // write non-superblock header

        // get size of participating block before
        size_t size_before = 0;
        bool found_size_before = false;
        for (int i = 1; i < WARP_SIZE - 1; i++) {
            size_t size_i_below = __shfl_up_sync(active_mask, size, i);
            // check if result valid. if not both threads are active and participating
            // in shuffle, then result is undefined
            if (!found_size_before && (my_lane_id - i >= 0) && is_active(my_lane_id - i, active_mask)) {
                size_before = size_i_below;
                found_size_before = true;
            }
        }
        assert(found_size_before);
        *header_ptr = size_before;
    }

    // indicate last block
    if (my_active_lane_id == n_threads - 1) {
        *header_ptr = *header_ptr | lastblock_bit_mask;
    }

    // make sure that no blocks are returned for which neighboring blocks are not setup
    // as this could lead to problems when the returned blocks are freed
    __syncwarp(active_mask);

    return (void*) (header_ptr + 1);
}

/*
 * If not last, do nothing. 
 * Otherwise, traverse allcoated blocks until 
 *	- find block that is not freed -> set to be last block
 *	- find superblock (that is free) -> call free
 */
__device__ void free_v4(void* memptr) {
    // check preconditions. If this is not given need rewrite bit manipulations
    assert(sizeof(size_t) == sizeof(void*));
    assert(sizeof(size_t) == sizeof(long long unsigned int)); // required for cast in CAS call
    size_t header_size = sizeof(void*);

    const size_t free_bit_mask = ((size_t) 1) << (header_size - 1);
    const size_t superblock_bit_mask = ((size_t) 1) << (header_size - 2);
    const size_t lastblock_bit_mask = ((size_t) 1) << (header_size - 3);
    const size_t size_mask = ~ (free_bit_mask | superblock_bit_mask | lastblock_bit_mask);

    size_t* header_ptr = ((size_t*) memptr) - 1;

    // set block to free
    *header_ptr = *header_ptr | free_bit_mask;

    if (!(*header_ptr & lastblock_bit_mask)) {
        return; // if we're not the last block, we're done
    }

    // from here on, we know that we have the last block
    // --> go through all prev blocks as described above

    size_t header = *header_ptr;
    do {
        do {
            // header ptr points to a freed block's header
            if (header & superblock_bit_mask) {
                // if we reach the superblock and it's free we're done
                free(header_ptr);
                return;
            }
            size_t size_prev_block = size_mask & header;
            header_ptr = (size_t*) (((char*) header_ptr) - header_size - size_prev_block);
            header = *header_ptr;
        } while (header & free_bit_mask);

        // reached a non-free block -> try to set it to last block if it has not been modified inbetween
        // note that modified = freed here as no other modifications possible
    } while (atomicCAS((long long unsigned int*) header_ptr, (long long unsigned int) header, (long long unsigned int) (header | lastblock_bit_mask)) != header);
    // if the above CAS fails, we know that the block header has been modified -> block freed, and we
    // will continue walking through the free blocks

    // once we exit this loop we succeeded in setting an earlier unfreed block to be the last block -> we're done
}
