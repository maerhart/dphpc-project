#include "common.h"

__global__ void memcpy_multithreaded(volatile void *dst, volatile void *src, size_t n) {
    size_t thread_idx = threadIdx.x + blockIdx.x * blockDim.x;

    volatile char *d = (volatile char*) dst;
    volatile char *s = (volatile char*) src;
    for (size_t i = thread_idx; i < n; i += blockDim.x * gridDim.x) {
        d[i] = s[i];
    }
}
