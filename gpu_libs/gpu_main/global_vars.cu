#include "global_vars.cuh"

#include "cuda_mpi.cuh"

namespace CudaMPI {

__device__ void* translateGlobalVar(const void* ptr, size_t size) {
    GlobalVarsStorage& gvs = CudaMPI::threadPrivateState().globalVarsStorage;
    return gvs.getValue(ptr, size);
}

} // namespace

