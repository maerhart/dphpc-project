#ifndef STDLIB_H_CUH
#define STDLIB_H_CUH

#include <stdlib.h>

#define qsort __gpu_qsort
__device__ void qsort(void *base, size_t nmemb, size_t size,
                  int (*compar)(const void *, const void *));

#define atoi __gpu_atoi
__device__ int atoi(const char *nptr);

#define atol __gpu_atol
__device__ long atol(const char *nptr);

#define realloc __gpu_realloc
__device__ void *realloc(void *ptr, size_t size);


#define srand __gpu_srand
__device__ void srand(unsigned int seed);

#define rand __gpu_rand
__device__ int rand(void);

#define getenv __gpu_getenv
__device__ char *getenv(const char *name);

#define exit __gpu_exit
__device__ void exit(int status);


#define strtod __gpu_strtod
__device__ double strtod(const char *nptr, char **endptr);

#define posix_memalign __posix_memalign
__device__ int posix_memalign(void **memptr, size_t alignment, size_t size);

#endif // STDLIB_H_CUH
