#include "omp.cuh"

__device__ int __gpu_omp_get_max_threads() {
    return 1;
}
