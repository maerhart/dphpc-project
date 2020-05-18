#include "stdlib.h.cuh"

#include "assert.h.cuh"

__device__ void __gpu_qsort(void *base, size_t nmemb, size_t size,
                      int (*compar)(const void *, const void *))
{
    NOT_IMPLEMENTED
}

__device__ void *__gpu_realloc(void *ptr, size_t size) {
    NOT_IMPLEMENTED
    return ptr;
}


__device__ void __gpu_srand(unsigned int seed) {
    NOT_IMPLEMENTED
}

__device__ int __gpu_rand(void) {
    NOT_IMPLEMENTED
    return 10;
}

__device__ char stub[] = "stub";

__device__ char *__gpu_getenv(const char *name) {
    NOT_IMPLEMENTED
    return stub;
}

__device__ void __gpu_exit(int) {
    asm("exit;");
}

__device__ void __gpu_abort() {
    asm("trap;");
}

__device__ int __gpu_posix_memalign(void **memptr, size_t alignment, size_t size) {
    NOT_IMPLEMENTED
    return 0;
}
