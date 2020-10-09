#ifndef STDLIB_CUH
#define STDLIB_CUH

#include <stdlib.h>

#include "assert.cuh"

#define __gpu_RAND_MAX INT_MAX

__device__ void __gpu_qsort(void *base, size_t nmemb, size_t size,
                  int (*compar)(const void *, const void *));

__device__ int __gpu_atoi(const char *nptr);

__device__ long __gpu_atol(const char *nptr);

__device__ void *__gpu_realloc(void *ptr, size_t size);

__device__ void __gpu_srand(unsigned int seed);

__device__ int __gpu_rand(void);

__device__ char *__gpu_getenv(const char *name);

__device__ void __gpu_exit(int status);

__device__ void __gpu_abort();

__device__ double __gpu_strtod(const char *nptr, char **endptr);

__device__ int __gpu_posix_memalign(void **memptr, size_t alignment, size_t size);

__device__ void* __gpu_malloc(size_t size);

__device__ void* __gpu_calloc(size_t nmemb, size_t size);

#endif
