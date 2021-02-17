#pragma once

namespace CudaMPI {
__device__ void* translateGlobalVar(const void* ptr, size_t size);
} // namespace

//template <typename T>
//constexpr __device__ T& __gpu_global(T& x) {
//    // If we are inside device code, then we use <pointer address, current rank> as a key
//    // in the internal mapping to actual value.
//    // We need to know size of type to get the correct offset in array of per-thread global variable copies.
//    return *(T*) CudaMPI::translateGlobalVar(&x, sizeof(x));
//}

#define __gpu_global(var) (var[threadIdx.x + blockDim.x * blockIdx.x])

