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
 * assuming block can start at given offset from 16-aligned starting point, compute where block will end
 *
 * @param offset Offset from 16-aligned starting point
 * @param min_header_size Minimum header size required
 * @param alignment Required Alignemnt of block payload
 */
__device__ size_t shift_offset(size_t offset, size_t min_header_size, size_t alignment) {
}

/**
 *  Warp level malloc with variable alignemnt and variable header size

 *  Each block has a header 
 *  where the first three  bits are is_superblock, is_free, and is_last_block
 *  and the remaining bits denote the size of the previous block
 *
 * 
 *  block 8-aligned <=> Header 8 byte
 *  block 4-aligned <=> Header 4 byte
 *  block 2-aligned <=> Header 2 byte
 *  block 1-aligned <=> Header 1 byte
 *
 */
__device__ void* malloc_v5(size_t size) {

    assert(sizeof(size_t) == 8);
    // check that size < max size which is 2 ^ (64 - 3) - 1 as need to fit size in header together with extra bits
    if (size & (((size_t) 7) << 61) || size < 1) {
        return NULL;
    }


    /*
    size_t header_size = sizeof(void*) * 8;

    const size_t free_bit_mask = ((size_t) 1) << (header_size - 1);
    const size_t superblock_bit_mask = ((size_t) 1) << (header_size - 2);
    const size_t lastblock_bit_mask = ((size_t) 1) << (header_size - 3);

    // assert special bits not used
    if ((free_bit_mask | superblock_bit_mask | lastblock_bit_mask) & size) {
        return NULL;
    }
    */

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

    // find out size of thread before
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
    assert(found_size_before || my_lane_id == elected_lane);

    // 2. check what header size we require
    // header has to be as big as alignment in order to conclude header size (or max of 8 bytes)
    // alignment has to as big as header in order to have legal position for header
    size_t min_header_size;
    size_t alignment;
    assert(sizeof(max_align_t) == 32);
    if (size < 2 && size_before < 32) { // 2 ^ (8-3)
        // can 1-align header as no 2-alignment required
        alignment = 1;
        // can fit in 1 byte / 8 bits together with header
        min_header_size = 1;
    } else if (size < 4 && size_before < 8192) { // 2 ^ (16 - 3)
        alignment = 2;
        min_header_size = 2;
    } else if (size < 8 && size_before < 536870912) { // 2 ^ (32 - 3)
        alignment = 4;
        min_header_size = 4;
    } else if (size < 16) { // know that size_before fits from initial check
        alignment = 8;
        min_header_size = 8;
    } else if (size < 32) {
        alignment = 16;
        min_header_size = 8; // header never bigger than 8
    } else {
        alignment = 32;
        min_header_size = 8;
    }

    // compute offset to all memory blocks before to know own offset
    size_t offset = 0;
    for (int i = WARP_SIZE - 1; i > 0; i--) {  // go through alignments and header sizes from bottom up
        size_t size_i_below = __shfl_up_sync(active_mask, size, i);
        size_t alignment_i_below = __shfl_up_sync(active_mask, alignment, i);
        size_t min_header_size_i_below = __shfl_up_sync(active_mask, min_header_size, i);
        // check if result valid. if not both threads are active and participating
        // in shuffle, then result is undefined
        if (!found_size_before && (my_lane_id - i >= 0) && is_active(my_lane_id - i, active_mask)) {
            // ensure alignment of header
            if (offset % min_header_size_i_below > 0) {
                offset += min_header_size_i_below - offset % min_header_size_i_below; 
            }

            // ensure that header aligned not by more than x if header size = x < 8 (necessary for free)
            if (min_header_size_i_below < 8 && offset % (2 * min_header_size_i_below) == 0) {
                offset += min_header_size_i_below; // now "misaligned" enough
            }

            // offset now at position where header of block i lanes below will start
            // (unless block alignment > 8 in which case header size 8 and we shift the start of the block but header is guaranteed to fit and aligned as required)
            offset += min_header_size_i_below;

            // add size of block, ensure that alignment of block satisfied
            if (offset % alignment_i_below > 0) {
                offset += alignment - offset % alignment;
            }

            // offset now at position where payload of block i lanes below will start
            offset += size_i_below;
        }
    }
    // offset contains offset from malloced ptr to where header of this block will start


    size_t total_superblock_length = 0;
    if (my_active_lane_id == n_threads - 1) {
        total_superblock_length = offset;
    }

    my_active_lane_id; //

    // TODO get required size from last thread to elected_lane and malloc
    // TODO identify payload pos and write header







    // ----------------

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

        // get size of participating block before TODO bug as elected lane does note participate
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
__device__ void free_v5(void* memptr) {
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

