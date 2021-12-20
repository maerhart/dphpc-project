__device__ void *malloc_v3(size_t size);

__device__ void free_v3(void *memptr);

__device__ void *malloc_v6(size_t size);

__device__ void free_v6(void *memptr);

__device__ void init_malloc();

__device__ void clean_malloc();
