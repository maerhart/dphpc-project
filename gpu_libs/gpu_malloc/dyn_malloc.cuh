
__device__ void *dyn_malloc(size_t size, bool coalesced=false);

__device__ void dyn_free(void *memptr);
