__device__ void *malloc_v3_comb(size_t size);

__device__ bool free_v3_comb(void *memptr);

__device__ void *malloc_v6(size_t size);

__device__ bool free_v6(void *memptr);

__device__ void init_malloc();

__device__ void clean_malloc();
