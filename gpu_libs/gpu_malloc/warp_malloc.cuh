
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
__device__ void* malloc_v5(size_t size, bool coalesced = false);


/*
 * If not last, do nothing. 
 * Otherwise, traverse allcoated blocks until 
 *	- find block that is not freed -> set to be last block
 *	- find superblock (that is free) -> call free
 */
__device__ void free_v5(void* memptr);

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
__device__ void* malloc_v4(size_t size, bool coalesced = false);

/*
 * If not last, do nothing. 
 * Otherwise, traverse allcoated blocks until 
 *	- find block that is not freed -> set to be last block
 *	- find superblock (that is free) -> call free
 */
__device__ void free_v4(void* memptr);
