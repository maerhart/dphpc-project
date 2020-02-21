#include "device_host_comm.cuh"

#include "cuda_mpi.cuh"

__device__ void* allocate_host_mem(size_t size) {
    return CudaMPI::sharedState().freeManagedMemory.allocate(size);
}

__device__ void free_host_mem(void* ptr) {
    CudaMPI::sharedState().freeManagedMemory.free(ptr);
}

__device__ void delegate_to_host(void* ptr, size_t size) {
    CudaMPI::sharedState().deviceToHostCommunicator.delegateToHost(ptr, size);
}
