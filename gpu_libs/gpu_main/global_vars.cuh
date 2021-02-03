#pragma once

// assigned from converter
extern __device__ int __gpu_num_globals;
extern __device__ void* __gpu_global_ptrs[];
extern __device__ int __gpu_global_size[];

// assigned from main.cu
extern __device__ void** __gpu_global_vars;

template <int Idx, typename T>
__device__ T& __gpu_global(T& x) {
    int rank = threadIdx.x + blockDim.x * blockIdx.x;
    return *(T*)(__gpu_global_vars[Idx + rank * __gpu_num_globals]);
}

template <int Idx, typename T>
__device__ void __gpu_init_global(T& x) {
    int rank = threadIdx.x + blockDim.x * blockIdx.x;
    void*& ptr = __gpu_global_vars[Idx + rank * __gpu_num_globals];
    if (!ptr) {
        ptr = malloc(sizeof(x));
        memcpy(ptr, &x, sizeof(x));
    }
}
