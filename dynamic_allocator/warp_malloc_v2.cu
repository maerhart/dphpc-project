#include <assert.h>
#include <iostream>
#include <utility>


__device__ const int WARP_SIZE = 32;
__device__ const size_t MAX_HEADER_PAD = sizeof(max_align_t) - 1;

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
    return lid >= 0 && lid < 32 && (((uint32_t) 1) << lid) & active_mask;
}

/**
 * Align x to given alignment by padding if necessary
 */
__device__ size_t pad_align(size_t x, size_t alignment) {
    size_t mod = x % alignment;
    if (mod > 0) {
       return x + alignment - mod; 
    } else {
        return x;
    }
}

/**
 * assuming header can start at given offset from max-aligned starting point, compute where header will end
 *
 * @param offset Offset from max-aligned starting point
 * @param min_header_size Minimum header size required
 * @param payload_alignment Required alignment for payload.
 *
 * Requires payload_alignment == min_header_size || min_header_size == 8 && payload_alignment % min_header_size == 0
 * and sizeof(size_t) == 8 
 *
 * @return offset to end of header (and start of payload)
 * 
 */
__device__ size_t header_end_offset(size_t offset, size_t min_header_size, size_t payload_alignment) {
    assert(payload_alignment == min_header_size || min_header_size == 8 && payload_alignment % min_header_size == 0);
    assert(sizeof(size_t) == 8);
    size_t offset_initial = offset;

    // ensure alignment of header
    offset = pad_align(offset, min_header_size);

    // ensure that header aligned not by more than x if header size = x < 8 (necessary for free)
    if (min_header_size < 8 && offset % (2 * min_header_size) == 0) {
        offset += min_header_size; // now "misaligned" enough
    }

    // offset now at position where header of block can start
    offset += min_header_size;

    // due to precondition this further padding will only happen in the case min_header_size = 8
    // and it will not destroy padding to min_header_size
    offset = pad_align(offset, payload_alignment);

    assert(offset - offset_initial <= MAX_HEADER_PAD);
    return offset;
}

/**
 * compute minimum required header size and required alignemnt for a block
 *
 * @param size Requested payload size for the block
 * @param space_prev_block Space required for payload of previous block (including padding)
 *          has to be < 2^(64 - 3)
 *
 * @param res_min_header_size Will contain min_header_size
 * @param res_alignment Will contain alignment
 */
__device__ void compute_min_header_size_alignment(size_t size, size_t space_prev_block_payload, size_t& res_min_header_size, size_t& res_alignment) {
    // header has to be as big as alignment in order to conclude header size (or max of 8 bytes)
    // alignment has to as big as header in order to have legal position for header
    assert(sizeof(max_align_t) == 32); // TODO not even cuda's malloc aligns to 32

    // upper bound on space required by previous block where we count the padding of this blocks header too
    size_t bound_space_prev_block = space_prev_block_payload + MAX_HEADER_PAD;

    /* if (size < 2 && bound_space_prev_block < 32) { // 2 ^ (8-3) // TODO cannot use atomicCAS on char and on short only with compute capability >= 7
        // can 1-align header as no 2-alignment required
        res_alignment = 1;
        // can fit in 1 byte / 8 bits together with header
        res_min_header_size = 1;
    } else if (size < 4 &&  bound_space_prev_block < 8192) { // 2 ^ (16 - 3)
        res_alignment = 2;
        res_min_header_size = 2;
    } else */ if (size < 8 &&  bound_space_prev_block < 536870912) { // 2 ^ (32 - 3)
        res_alignment = 4;
        res_min_header_size = 4;
    } else if (size < 16) { // know that space_prev_block fits from initial check
        res_alignment = 8;
        res_min_header_size = 8;
    } else { // if (size < 32) { // TODO align to 32? not even cuda's malloc aligns to 32 but max_align_t is 32
        res_alignment = 16;
        res_min_header_size = 8; // header never bigger than 8
    } /*else {
        res_alignment = 32;
        res_min_header_size = 8;
    }*/
}

/**
 * write the header for a block.
 */
template<typename T>
__device__ void write_header(void* payload_start_ptr, bool is_superblock, bool is_lastblock, void* prev_payload_start_ptr) {
   T* header_ptr = ((T*) payload_start_ptr) - 1;
   size_t space_prev_payload = ((char*) header_ptr) - ((char*) prev_payload_start_ptr);  // includes padding

   T header = (T) space_prev_payload;

   assert(header == space_prev_payload);

   // write superblock bit
   header = header | (((T) is_superblock) << (8 * sizeof(T) - 1));

   // write lastblock bit
   header = header | (((T) is_lastblock) << (8 * sizeof(T) - 3));

   *header_ptr = header;
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

    int my_lane_id = lane_id();

    // retrieve mask of all threads in this warp that are currently executing
    // this instruction. they will perform a coalesced malloc
    uint32_t active_mask = __activemask();
    // count number of 1s
    int n_threads = __popc(active_mask);
    // Find the lowest-numbered active lane
    int elected_lane = __ffs(active_mask) - 1;
    bool is_elected = my_lane_id == elected_lane;
    // Find the highest-numbered active lane
    int last_lane = 31 - __clz(active_mask);
    bool is_last = my_lane_id == last_lane;
    // get id/idx among active lanes
    int my_active_lane_id = active_lane_id(active_mask);

    // compute relevant offsets from 16-bit aligned malloced superblock start
    size_t offset_prev_payload_end = 0; // offset to end of payload of last processed block;
    size_t offset_prev_payload_start = 0; // offset to end of header of last processed block (or start of payload, equivalent)
    for (int i = WARP_SIZE - 1; i > 0; i--) {  // go through all lanes/their memory  blocks from lowest lane to highest
        size_t size_i_below = __shfl_up_sync(active_mask, size, i);
        // check if result valid. if not both threads are active and participating
        // in shuffle, then result is undefined
        if (is_active(my_lane_id - i, active_mask)) {
            size_t min_header_size; size_t alignment;
            compute_min_header_size_alignment(size_i_below, offset_prev_payload_end - offset_prev_payload_start, min_header_size, alignment);
            offset_prev_payload_start = header_end_offset(offset_prev_payload_end, min_header_size, alignment);
            offset_prev_payload_end = offset_prev_payload_start + size;
        }
    }
    // arrived at own block, offset vars contain block of lane before

    // compute minimum header size and payload alignment for this block
    size_t min_header_size; size_t alignment;
    compute_min_header_size_alignment(size, offset_prev_payload_end - offset_prev_payload_start, min_header_size, alignment);
    size_t offset_payload_start = header_end_offset(offset_prev_payload_end, min_header_size, alignment);

    // let last thread compute required total length (= offset to end of its payload)
    size_t total_superblock_length = 0;
    if (is_last) {
        total_superblock_length = offset_payload_start + size;
    }
    total_superblock_length = __shfl_sync(active_mask, total_superblock_length, last_lane);


    // perform malloc of superblock
    char* malloced_ptr = NULL;
    // perform coalesced malloc
    if (is_elected) {
        malloced_ptr = (char*) malloc(total_superblock_length);
    }
    // broadcast alloced ptr to all lanes
    assert(sizeof(size_t) == sizeof(char*)); // make sure we don't change due to cast
    // need to cast as pointers can't be shuffled
    malloced_ptr = (char*) __shfl_sync(active_mask, (size_t) malloced_ptr, elected_lane);

    void* payload_start_ptr = malloced_ptr + offset_payload_start;
    void* prev_payload_start_ptr = malloced_ptr + offset_prev_payload_start;

    size_t payload_start_num = (size_t) payload_start_ptr;
    // work with correct header type
    if (payload_start_num % 8 == 0) {
        write_header<size_t>(payload_start_ptr, is_elected, is_last, prev_payload_start_ptr);
    } else if (payload_start_num % 4 == 0) {
        write_header<uint32_t>(payload_start_ptr, is_elected, is_last, prev_payload_start_ptr);
    } else {
        write_header<uint16_t>(payload_start_ptr, is_elected, is_last, prev_payload_start_ptr);
    }

    __syncwarp(active_mask); // required s.t. not uninitialized headers are looked at during free
    return payload_start_ptr;
}


typedef unsigned int min_h_t; // uint32_t header are the smallest headers we can use in compute capability 6

template<typename T>
__device__ min_h_t* read_header_templ(char* payload_start_ptr, size_t& size_result) {
   T* header_ptr = ((T*) payload_start_ptr) - 1;

   // read size
   size_result = (size_t) (*header_ptr & ~(((T) 7) << (8 * sizeof(T) - 3)));

   return (min_h_t*) header_ptr;
}


/**
 * read the header of a block
 * @param payload_start_ptr Ptr to start of payload
 * @param size_result Will hold size of prev block stored in header
 *
 * @return Pointer to start of header
 */
__device__ min_h_t* read_header(char* payload_start_ptr, size_t& size_result) {
    size_t payload_start_num = (size_t) payload_start_ptr;
    // work with correct header type
    if (payload_start_num % 8 == 0) {
        return read_header_templ<size_t>(payload_start_ptr, size_result);
    } else {// if (payload_start_num % 4 == 0) {
        return read_header_templ<uint32_t>(payload_start_ptr, size_result);
    } /*else { currently not supported
        return read_header_templ<uint16_t>(payload_start_ptr, size_result);
    }*/
}

/*
 * If not last, do nothing. 
 * Otherwise, traverse allcoated blocks until 
 *	- find block that is not freed -> set to be last block
 *	- find superblock (that is free) -> call free
 */
__device__ void free_v5(void* memptr) {

    min_h_t superblock_bit_mask = ((min_h_t) 1) << sizeof(min_h_t) - 1;
    min_h_t free_bit_mask = ((min_h_t) 1) << sizeof(min_h_t) - 2;
    min_h_t lastblock_bit_mask = ((min_h_t) 1) << sizeof(min_h_t) - 3;

    char* payload_start_ptr = (char*) memptr;
    size_t size_prev_block;
    min_h_t* header_start = read_header(payload_start_ptr, size_prev_block); // points to start of header (header might be larger than 16 bits)

    // mark block as free
    *header_start = *header_start | free_bit_mask;

    if (!(*header_start & lastblock_bit_mask)) {
        // block is not last block -> done (only last block does work
        return;
    }

    printf("In free thread %d\n", threadIdx.x);

    // from here on, we know that we have the last block
    // --> go through all prev blocks as described above

    min_h_t header_bits = *header_start;
    do {
        do {
            // payload_start_ptr, header_start, and header_bits contain freed block's info
            if (header_bits & superblock_bit_mask) {
                // if we reach the superblock and it's free we're done
                free(header_start);
                return;
            }
            // look at block before
            payload_start_ptr  = (((char*) header_start) - size_prev_block);
            min_h_t* header_start = read_header(payload_start_ptr, size_prev_block);
            header_bits = *header_start;
        } while (header_bits & free_bit_mask);

        // reached a non-free block -> try to set it to last block if it has not been modified inbetween
        // note that modified = freed here as no other modifications possible
    } while (atomicCAS(header_start, header_bits, (header_bits | lastblock_bit_mask)) != header_bits);
    // if the above CAS fails, we know that the block header has been modified -> block freed, and we
    // will continue walking through the free blocks

    // once we exit this loop we succeeded in setting an earlier unfreed block to be the last block -> we're done
}

