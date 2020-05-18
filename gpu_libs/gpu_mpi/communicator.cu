#include "communicator.cuh"

#include "mpi_common.cuh"
#include "group.cuh"

#include <cooperative_groups.h>
using namespace cooperative_groups;

#include "cuda_mpi.cuh"

#include "mpi.h.cuh"

#include "assert.h.cuh"

#include "stdlib.cuh"

struct MPI_Comm_impl {
    int context;
    MPI_Group group;
};


__device__ MPI_Comm MPI_COMM_WORLD = (MPI_Comm)nullptr;
__device__ MPI_Comm MPI_COMM_NULL = (MPI_Comm)nullptr;

namespace gpu_mpi {

__device__ int createNewContextId(MPI_Comm comm) {
    int freeCommCtx = CudaMPI::threadPrivateState().unusedCommunicationContext;
    MPI_Allreduce(&freeCommCtx, &freeCommCtx, 1, MPI_INT, MPI_MAX, comm);
    CudaMPI::threadPrivateState().unusedCommunicationContext = freeCommCtx + 1;
    if (freeCommCtx > (1 << 15)) {
        printf("ERROR: the limit of communicator contexts is exceeded!\n");
        __gpu_abort();
    }
    return freeCommCtx;
}
    
__device__ int getCommContext(MPI_Comm comm) {
    return comm->context;
};

__device__ void initializeGlobalCommunicators() {
    if (this_grid().thread_rank() == 0) {
        MPI_COMM_NULL = new MPI_Comm_impl;
        MPI_COMM_NULL->context = 0;
        MPI_COMM_NULL->group = MPI_GROUP_EMPTY;
        
        MPI_COMM_WORLD = new MPI_Comm_impl;
        MPI_COMM_WORLD->context = 1;
        MPI_COMM_WORLD->group = MPI_GROUP_WORLD;
        
        CudaMPI::threadPrivateState().unusedCommunicationContext = 2;
    }
    this_grid().sync();
}

} // namespace

__device__ int MPI_Comm_free(MPI_Comm *comm) {
    delete *comm;
    return MPI_SUCCESS;
}

__device__ int MPI_Comm_group(MPI_Comm comm, MPI_Group *group) {
    *group = comm->group;
    return MPI_SUCCESS;
}

__device__ int MPI_Comm_create(
    MPI_Comm comm, MPI_Group group, MPI_Comm *newcomm) 
{
    int ctxId = gpu_mpi::createNewContextId(comm);
    
    int rank = MPI_UNDEFINED;
    MPI_Group_rank(group, &rank);
    if (rank == MPI_UNDEFINED) {
        *newcomm = MPI_COMM_NULL;
        return MPI_SUCCESS;
    } else {
        MPI_Comm commImpl = new MPI_Comm_impl;
        commImpl->context = ctxId;
        commImpl->group = group;
        *newcomm = commImpl;
        return MPI_SUCCESS;
    }
}

__device__ int MPI_Attr_put(MPI_Comm comm, int keyval, void *attribute_val) {
    return MPI_SUCCESS;
}

__device__ int MPI_Attr_get(MPI_Comm comm, int keyval,void *attribute_val, int *flag) {
    return MPI_SUCCESS;
}

__device__ int MPI_Cart_create(MPI_Comm comm_old, int ndims, const int dims[], const int periods[], int reorder, MPI_Comm *comm_cart) {
    MPI_Comm commImpl = new MPI_Comm_impl;
    
    int ctxId = gpu_mpi::createNewContextId(comm_old);
    
    commImpl->context = ctxId;
    commImpl->group = comm_old->group;
    
    *comm_cart = commImpl;
    
    return MPI_SUCCESS;
}

__device__ int MPI_Cart_sub(MPI_Comm comm, const int remain_dims[], MPI_Comm *comm_new) {
    NOT_IMPLEMENTED
    return MPI_SUCCESS;
}

__device__ int MPI_Comm_split(MPI_Comm comm, int color, int key, MPI_Comm *newcomm) {
    NOT_IMPLEMENTED
    return MPI_SUCCESS;
}

__device__ int MPI_Comm_size(MPI_Comm comm, int *size) {
    return MPI_Group_size(comm->group, size);
}

__device__ int MPI_Comm_rank(MPI_Comm comm, int *rank) {
    return MPI_Group_rank(comm->group, rank);
}


