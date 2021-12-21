
// memory layout of a superblock
// s_header, blocksize x [pointer to s_header, data]
__device__ void* malloc_v1(size_t size, bool coalesced = false);

__device__ void free_v1(void* memptr);

__device__ void* malloc_v2(size_t size, bool coalesced = false);

__device__ void free_v2(void* memptr);

__device__ void init_malloc_v3();

__device__ void clean_malloc_v3();

__device__ void* malloc_v3(size_t size, bool coalesced = false);

__device__ void free_v3(void *memptr);

__device__ void* malloc_v6(size_t size, bool coalesced = false);

__device__ void free_v6(void *memptr);
