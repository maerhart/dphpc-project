#ifndef Alloc_H
#define Alloc_H
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>

#ifdef USE_GPU
#include <cuda_runtime.h>

__host__ __device__ inline long get_idx(long v, long w, long x, long y, long z,
                                        long stride_w, long stride_x,
                                        long stride_y, long stride_z) {
  return stride_x * stride_y * stride_z * w + stride_y * stride_z * x +
         stride_z * y + z;
}

__host__ __device__ inline long get_idx(long w, long x, long y, long z,
                                        long stride_x, long stride_y,
                                        long stride_z) {
  return stride_x * stride_y * stride_z * w + stride_y * stride_z * x +
         stride_z * y + z;
}

__host__ __device__ inline long get_idx(long x, long y, long z, long stride_y,
                                        long stride_z) {
  return stride_y * stride_z * x + stride_z * y + z;
}

__host__ __device__ inline long get_idx(long x, long y, long s1) {
  return x + (y * s1);
}
#endif

/**
 * Create a d dimensional pointer hierarchy array
 */
void *newArr(int typesize, int dim, ...);

/**
 * Create hierarchical pointer structure to mimic multidimensional arrays, 
 * where data is dynamically allocated on the heap, and data array is 
 * contigous in memory to allow efficient communication of data. 
 */
void **ptrArr(void **in, int typesize, int dim, ...);

/**
 * Deallocate pointer array
 */
void delArr(int dim, void *arr);


#endif
