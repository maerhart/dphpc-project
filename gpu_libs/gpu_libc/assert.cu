#include "assert.cuh"
#include "string.cuh"
#include "common.h"

struct GPU_MPI_Error {
    static const int max_msg_size = 1024;
    char msg[max_msg_size];
    int occured;
    int bx, by, bz, tx, ty, tz;
    
    __host__ __device__ void reset() {
        occured = 0;
        bx = by = bz = tx = ty = tz = -1;
        msg[0] = '\0';
    }
};

static GPU_MPI_Error* host_err;
static __device__ GPU_MPI_Error* gpu_mpi_err;
static __device__ int err_lock;

namespace CudaMPI {

void initError() {
    host_err = nullptr;
    GPU_MPI_Error* device_err = nullptr;
    CUDA_CHECK(cudaHostAlloc(&host_err, sizeof(GPU_MPI_Error), cudaHostAllocMapped | cudaHostAllocPortable));
    CUDA_CHECK(cudaHostGetDevicePointer(&device_err, host_err, /* unused flags */ 0));
    CUDA_CHECK(cudaMemcpyToSymbol(gpu_mpi_err, &device_err, sizeof(GPU_MPI_Error*)));
}

void freeError() {
    CUDA_CHECK(cudaFreeHost(host_err));
    host_err = nullptr;
}

__device__ void recordError(const char* msg) {
    if (atomicCAS(&err_lock, 0, 1)) {
        return; // another thread already wrote into err
    }

    GPU_MPI_Error& e = *gpu_mpi_err;

    e.occured = 1;

    __gpu_strncpy(e.msg, msg, GPU_MPI_Error::max_msg_size - 1);

    e.bx = blockIdx.x;
    e.by = blockIdx.y;
    e.bz = blockIdx.z;
    e.tx = threadIdx.x;
    e.ty = threadIdx.y;
    e.tz = threadIdx.z;

    __threadfence_system();
}

void printLastError() {
    if (!host_err) {
        printf("printLastError(): ERROR host pointer is not initialized!\n");
        return;
    }
    GPU_MPI_Error& err = *host_err;
    if (err.occured) {
        printf("GPUMPI: %s block <%d,%d,%d> thread <%d,%d,%d>\n",
            err.msg, err.bx, err.by, err.bz, err.tx, err.ty, err.tz
        );
    } else {
        printf("GPUMPI: no errors occured\n");
    }
}

} // namespace