#include "stdlib.cuh"

#include "assert.cuh"
#include "cuda_mpi.cuh"
#include "dyn_malloc.cuh"

__device__ void __gpu_qsort(void *base, size_t nmemb, size_t size,
                      int (*compar)(const void *, const void *))
{
    char* ptr = (char*)base;
    for (int i = 0; i < nmemb; i++) {
        for (int j = i + 1; j < nmemb; j++) {
            if (compar(ptr + i * size, ptr + j * size) > 0) {
                for (int s = 0; s < size; s++) {
                    char tmp = ptr[i * size + s];
                    ptr[i * size + s] = ptr[j * size + s];
                    ptr[j * size + s] = tmp;
                }
            }
        }
    }
}

__device__ void *__gpu_realloc(void *ptr, size_t size) {
    NOT_IMPLEMENTED;
    return ptr;
}


__device__ void __gpu_srand(unsigned int seed) {
    curand_init(seed, 0, 0, &CudaMPI::threadPrivateState().rand_state);
}

__device__ int __gpu_rand(void) {
    unsigned int uval = curand(&CudaMPI::threadPrivateState().rand_state);
    // use 31 bits of randomness instead of 32 bits
    // and make sure that result in [0, __gpu_RAND_MAX]
    int val = uval / 2; 
    assert(0 <= val && val <= __gpu_RAND_MAX);
    return val;
}

__device__ char stub[] = "stub";

__device__ char *__gpu_getenv(const char *name) {
    // TODO return proper thing instead
    // for now pretend that there is no env vars
    return nullptr;
}

__device__ void __gpu_exit(int) {
    printf("GPUMPI: DEVICE-SIDE EXIT\n");
    asm("exit;");
}

__device__ void __gpu_abort() {
    printf("GPUMPI: DEVICE-SIDE ABORT\n");
    asm("trap;");
}

__device__ int __gpu_posix_memalign(void **memptr, size_t alignment, size_t size) {
    NOT_IMPLEMENTED;
    return 0;
}

__device__ void* __gpu_malloc(size_t size) {
    void* ptr = malloc(size);
    #ifndef NDEBUG
    printf("Performed standard malloc\n");
    if (!ptr) {
        printf("GPUMPI: malloc failed to allocate %llu bytes on device\n", (long long unsigned)size);
    }
    #endif
    return ptr;
}

__device__ void* __gpu_calloc(size_t nmemb, size_t size, bool coalesced) {
    void* ptr = __gpu_malloc(nmemb * size, coalesced);
    if (ptr) {
        memset(ptr, 0, nmemb * size);
    }
    return ptr;
}

__device__ void __gpu_free(void *memptr) {
    free(memptr);
}

__device__ void* __gpu_malloc_coalesce(size_t size, bool coalesced) {   
    void* ptr = dyn_malloc(size, coalesced);
    #ifndef NDEBUG
    printf("Performed dyn_malloc\n");
    if (!ptr) {
        printf("GPUMPI: malloc failed to allocate %llu bytes on device\n", (long long unsigned)size);
    }
    #endif
    return ptr;
}

__device__ void* __gpu_calloc_coalesce(size_t nmemb, size_t size, bool coalesced) {
    void* ptr = __gpu_malloc_coalesce(nmemb * size, coalesced);
    if (ptr) {
        memset(ptr, 0, nmemb * size);
    }
    return ptr;
}

__device__ void __gpu_free_coalesce(void *memptr) {
    dyn_free(memptr);
}

__device__ void __gpu_init_malloc() {}

__device__ void __gpu_clean_malloc() {}
