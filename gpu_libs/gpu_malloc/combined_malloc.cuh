__device__ void *combined_malloc(size_t size, bool coalesced);

__device__ void combined_free(void *memptr);

__device__ void init_malloc();

__device__ void clean_malloc();