#include "stdlib.cuh"
#include "stdio.cuh"

#include "assert.cuh"
#include "cuda_mpi.cuh"
#include "dyn_malloc.cuh"
#include "warp_malloc.cuh"
#include "dynamic_allocator.cuh"
#include "combined_malloc.cuh"

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
    __gpu_fprintf(__gpu_stderr, "Performed standard malloc\n");
    if (!ptr) {
        printf("GPUMPI: malloc failed to allocate %llu bytes on device\n", (long long unsigned)size);
    }
    #endif
    return ptr;
}

__device__ void* __gpu_calloc(size_t nmemb, size_t size) {
    void* ptr = __gpu_malloc(nmemb * size);
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
    __gpu_fprintf(__gpu_stderr, "Performed dyn_malloc\n");
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

__device__ void* __gpu_malloc_v1(size_t size, bool coalesced) {   
    void* ptr = malloc_v1(size, coalesced);
    #ifndef NDEBUG
    __gpu_fprintf(__gpu_stderr, "Performed dyn_malloc\n");
    if (!ptr) {
        printf("GPUMPI: malloc failed to allocate %llu bytes on device\n", (long long unsigned)size);
    }
    #endif
    return ptr;
}

__device__ void* __gpu_calloc_v1(size_t nmemb, size_t size, bool coalesced) {
    void* ptr = __gpu_malloc_v1(nmemb * size, coalesced);
    if (ptr) {
        memset(ptr, 0, nmemb * size);
    }
    return ptr;
}

__device__ void __gpu_free_v1(void *memptr) {
    free_v1(memptr);
}

__device__ void* __gpu_malloc_v2(size_t size, bool coalesced) {   
    void* ptr = malloc_v2(size, coalesced);
    #ifndef NDEBUG
    __gpu_fprintf(__gpu_stderr, "Performed dyn_malloc\n");
    if (!ptr) {
        printf("GPUMPI: malloc failed to allocate %llu bytes on device\n", (long long unsigned)size);
    }
    #endif
    return ptr;
}

__device__ void* __gpu_calloc_v2(size_t nmemb, size_t size, bool coalesced) {
    void* ptr = __gpu_malloc_v2(nmemb * size, coalesced);
    if (ptr) {
        memset(ptr, 0, nmemb * size);
    }
    return ptr;
}

__device__ void __gpu_free_v2(void *memptr) {
    free_v2(memptr);
}

__device__ void* __gpu_malloc_v3(size_t size, bool coalesced) {   
    void* ptr = malloc_v3(size, coalesced);
    #ifndef NDEBUG
    __gpu_fprintf(__gpu_stderr, "Performed malloc_v3\n");
    if (!ptr) {
        printf("GPUMPI: malloc failed to allocate %llu bytes on device\n", (long long unsigned)size);
    }
    #endif
    return ptr;
}

__device__ void* __gpu_calloc_v3(size_t nmemb, size_t size, bool coalesced) {
    void* ptr = __gpu_malloc_v3(nmemb * size, coalesced);
    if (ptr) {
        memset(ptr, 0, nmemb * size);
    }
    return ptr;
}

__device__ void __gpu_free_v3(void *memptr) {
    free_v3(memptr);
}

__device__ void __gpu_init_malloc() {
    init_malloc_v3();
}

__device__ void __gpu_clean_malloc() {
    clean_malloc_v3();
}


__device__ void* __gpu_malloc_v4(size_t size, bool coalesced) {   
    void* ptr = malloc_v4(size, coalesced);
    #ifndef NDEBUG
    __gpu_fprintf(__gpu_stderr, "Performed malloc_v4\n");
    if (!ptr) {
        printf("GPUMPI: malloc failed to allocate %llu bytes on device\n", (long long unsigned)size);
    }
    #endif
    return ptr;
}

__device__ void* __gpu_calloc_v4(size_t nmemb, size_t size, bool coalesced) {
    void* ptr = __gpu_malloc_v4(nmemb * size, coalesced);
    if (ptr) {
        memset(ptr, 0, nmemb * size);
    }
    return ptr;
}

__device__ void __gpu_free_v4(void *memptr) {
    free_v4(memptr);
}

__device__ void* __gpu_malloc_v5(size_t size, bool coalesced) {   
    void* ptr = malloc_v5(size, coalesced);
    #ifndef NDEBUG
    __gpu_fprintf(__gpu_stderr, "Performed malloc_v5\n");
    if (!ptr) {
        printf("GPUMPI: malloc failed to allocate %llu bytes on device\n", (long long unsigned)size);
    }
    #endif
    return ptr;
}

__device__ void* __gpu_calloc_v5(size_t nmemb, size_t size, bool coalesced) {
    void* ptr = __gpu_malloc_v5(nmemb * size, coalesced);
    if (ptr) {
        memset(ptr, 0, nmemb * size);
    }
    return ptr;
}

__device__ void __gpu_free_v5(void *memptr) {
    free_v5(memptr);
}

__device__ void* __gpu_malloc_v6(size_t size, bool coalesced) {   
    void* ptr = combined_malloc(size, coalesced);
    #ifndef NDEBUG
    __gpu_fprintf(__gpu_stderr, "Performed malloc_v6\n");
    if (!ptr) {
        printf("GPUMPI: malloc failed to allocate %llu bytes on device\n", (long long unsigned)size);
    }
    #endif
    return ptr;
}

__device__ void* __gpu_calloc_v6(size_t nmemb, size_t size, bool coalesced) {
    void* ptr = __gpu_malloc_v6(nmemb * size, coalesced);
    if (ptr) {
        memset(ptr, 0, nmemb * size);
    }
    return ptr;
}

__device__ void __gpu_free_v6(void *memptr) {
    combined_free(memptr);
}

__device__ void __gpu_init_malloc_v6() {
    init_malloc();
}

__device__ void __gpu_clean_malloc_v6() {
    clean_malloc();
}