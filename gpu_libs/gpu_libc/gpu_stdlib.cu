#include "stdlib.h.cuh"

__device__ void qsort(void *base, size_t nmemb, size_t size,
                      int (*compar)(const void *, const void *)) {}

__device__ int atoi(const char *nptr) { return 0; }

__device__ long atol(const char *nptr) { return 0; }

__device__ void *realloc(void *ptr, size_t size) { return ptr; }


__device__ void srand(unsigned int seed) { return; }

__device__ int rand(void) {return 10; }

__device__ char stub[] = "stub";

__device__ char *getenv(const char *name) { return stub; }

__device__ void exit(int status) {}


__device__ double strtod(const char *nptr, char **endptr) { return 0.0; }

__device__ int posix_memalign(void **memptr, size_t alignment, size_t size) { return 0; }
