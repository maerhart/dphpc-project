#include "mpi.h.cuh"

// cuda_mpi.cuh should be included before device specific standard library functions
// because it relies on standard ones
#include "cuda_mpi.cuh"

#include "stdlib.h.cuh"
#include "string.h.cuh"

#include <cooperative_groups.h>
using namespace cooperative_groups;

#include "mpi_common.cuh"

#define MPI_COLLECTIVE_TAG (-2)

__device__ int MPI_Init(int *argc, char ***argv) {
    gpu_mpi::initializeGlobalGroups();
    gpu_mpi::initializeGlobalCommunicators();
    return MPI_SUCCESS;
}

__device__ int MPI_Finalize(void) {
    // TODO: due to exit() you need to perform
    // all MPI related memory deallocation here

    // notify host that there will be no messages from this thread anymore
    CudaMPI::sharedState().deviceToHostCommunicator.delegateToHost(0, 0);

    gpu_mpi::destroyGlobalGroups();
    gpu_mpi::destroyGlobalCommunicators();
    
    return MPI_SUCCESS;
}

__device__ int MPI_Get_processor_name(char *name, int *resultlen) {
    const char hardcoded_name[] = "GPU thread";
    strcpy(name, hardcoded_name);
    *resultlen = sizeof(hardcoded_name);
    return MPI_SUCCESS;
}

__device__ int MPI_Bcast(void *buffer, int count, MPI_Datatype datatype,
                         int root, MPI_Comm comm)
{
    int dataSize = gpu_mpi::plainTypeSize(datatype);
    assert(dataSize > 0);
    
    int commSize = -1;
    int commRank = -1;
    
    MPI_Comm_size(comm, &commSize);
    MPI_Comm_rank(comm, &commRank);
    
    int tag = MPI_COLLECTIVE_TAG;
    int ctx = gpu_mpi::getCommContext(comm);
    
    if (commRank == root) {
        CudaMPI::PendingOperation** ops = (CudaMPI::PendingOperation**) malloc(sizeof(CudaMPI::PendingOperation*) * commSize);
        assert(ops);
        for (int dst = 0; dst < commSize; dst++) {
            if (dst != commRank) {
                ops[dst] = CudaMPI::isend(dst, buffer, dataSize, ctx, tag);
            }
        }
        for (int dst = 0; dst < commSize; dst++) {
            if (dst != commRank) {
                CudaMPI::wait(ops[dst]);
            }
        }
        free(ops);
    } else {
        CudaMPI::PendingOperation* op = CudaMPI::irecv(root, buffer, dataSize, ctx, tag);
        CudaMPI::wait(op);
    }
    
    return MPI_SUCCESS;
}

__device__ double MPI_Wtime(void) {
    auto clock = clock64();
    double seconds = clock * MPI_Wtick();
    return seconds;
}

__device__ int MPI_Reduce(const void *sendbuf, void *recvbuf, int count,
                          MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm)
{
    int commSize = -1;
    int commRank = -1;
    MPI_Comm_size(comm, &commSize);
    MPI_Comm_rank(comm, &commRank);

    int dataSize = gpu_mpi::plainTypeSize(datatype) * count;
    assert(dataSize > 0);
    
    int tag = MPI_COLLECTIVE_TAG;
    int ctx = gpu_mpi::getCommContext(comm);
    
    if (commRank == root) {
        auto ops = (CudaMPI::PendingOperation**) malloc(sizeof(CudaMPI::PendingOperation*) * commSize);
        double* buffers = (double*) malloc(dataSize * commSize);
        assert(ops);
        for (int src = 0; src < commSize; src++) {
            if (src != commRank) {
                ops[src] = CudaMPI::irecv(src, buffers + src * count, dataSize, ctx, tag);
            }
        }
        for (int i = 0; i < count; i++) {
            assert(op == MPI_SUM);
            double* recvBufDouble = (double*) recvbuf;
            recvBufDouble[i] = 0;
        }
        for (int src = 0; src < commSize; src++) {
            double* tempBufDouble = nullptr;
            if (src != commRank) {
                CudaMPI::wait(ops[src]);
                tempBufDouble = buffers + src * count;
            } else {
                tempBufDouble = (double*) sendbuf;
            }
            double* recvBufDouble = (double*) recvbuf;
            
            for (int i = 0; i < count; i++) {
                assert(op == MPI_SUM);
                recvBufDouble[i] += tempBufDouble[i];
            }
        }
        
        free(buffers);
        free(ops);
    } else {
        CudaMPI::PendingOperation* op = CudaMPI::isend(root, sendbuf, dataSize, ctx, tag);
        CudaMPI::wait(op);
    }
    
    return MPI_SUCCESS;
}

__device__ int MPI_Type_contiguous(int count, MPI_Datatype oldtype, MPI_Datatype *newtype) {
    return MPI_SUCCESS;
}

__device__ int MPI_Type_commit(MPI_Datatype *datatype) {
    return MPI_SUCCESS;
}

__device__ int MPI_Recv(void *buf, int count, MPI_Datatype datatype,
                        int source, int tag, MPI_Comm comm, MPI_Status *status) {
    return MPI_SUCCESS;
}

__device__ int MPI_Sendrecv(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
            int dest, int sendtag, void *recvbuf, int recvcount,
            MPI_Datatype recvtype, int source, int recvtag,
                 MPI_Comm comm, MPI_Status *status) {
    return MPI_SUCCESS;
}

__device__ int MPI_Send(const void *buf, int count, MPI_Datatype datatype, int dest,
            int tag, MPI_Comm comm)
{
    return MPI_SUCCESS;
}

__device__ double MPI_Wtick() {
    int peakClockKHz = CudaMPI::threadPrivateState().peakClockKHz;
    return 0.001 / peakClockKHz;
}

__device__ int MPI_Allreduce(const void *sendbuf, void *recvbuf, int count,
                         MPI_Datatype datatype, MPI_Op op, MPI_Comm comm) {
    return MPI_SUCCESS;
}
__device__ int MPI_Abort(MPI_Comm comm, int errorcode) {
    return MPI_SUCCESS;
}
__device__ int MPI_Type_size(MPI_Datatype datatype, int *size) {
    return MPI_SUCCESS;
}
__device__ int MPI_Gather(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
            void *recvbuf, int recvcount, MPI_Datatype recvtype, int root,
            MPI_Comm comm) {
    return MPI_SUCCESS;
}

__device__ int MPI_Barrier(MPI_Comm comm) {
    return MPI_SUCCESS;
}
__device__ int MPI_Alltoall(const void *sendbuf, int sendcount,
            MPI_Datatype sendtype, void *recvbuf, int recvcount,
            MPI_Datatype recvtype, MPI_Comm comm) {
    return MPI_SUCCESS;
}
__device__ int MPI_Alltoallv(const void *sendbuf, const int sendcounts[],
            const int sdispls[], MPI_Datatype sendtype,
            void *recvbuf, const int recvcounts[],
            const int rdispls[], MPI_Datatype recvtype, MPI_Comm comm) {
    return MPI_SUCCESS;
}

__device__ int MPI_Allgather(const void *sendbuf, int  sendcount,
             MPI_Datatype sendtype, void *recvbuf, int recvcount,
             MPI_Datatype recvtype, MPI_Comm comm)
{
    NOT_IMPLEMENTED
    return MPI_SUCCESS;
}

__device__ int MPI_Allgatherv(const void *sendbuf, int sendcount,
                              MPI_Datatype sendtype, void *recvbuf, const int recvcounts[],
                              const int displs[], MPI_Datatype recvtype, MPI_Comm comm)
{
    NOT_IMPLEMENTED
    return MPI_SUCCESS;
}

__device__ int MPI_Gatherv(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                           void *recvbuf, const int recvcounts[], const int displs[], MPI_Datatype recvtype,
                           int root, MPI_Comm comm) {
    return MPI_SUCCESS;
}
__device__ int MPI_Scatter(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                           void *recvbuf, int recvcount, MPI_Datatype recvtype, int root,
                           MPI_Comm comm) {
    return MPI_SUCCESS;
}

__device__ int MPI_NULL_COPY_FN(MPI_Comm oldcomm, int keyval,
                     void *extra_state, void *attribute_val_in,
                     void *attribute_val_out, int *flag) {
    return MPI_SUCCESS;
}

__device__ int MPI_NULL_DELETE_FN(MPI_Comm comm, int keyval,
                       void *attribute_val, void *extra_state) {
    return MPI_SUCCESS;
}

__device__ int MPI_Keyval_create(MPI_Copy_function *copy_fn,
                                 MPI_Delete_function *delete_fn, int *keyval, void *extra_state) {
    return MPI_SUCCESS;
}

__device__ int MPI_Dims_create(int nnodes, int ndims, int dims[]) {
    return MPI_SUCCESS;
}

__device__ int MPI_Irecv(void *buf, int count, MPI_Datatype datatype,
               int source, int tag, MPI_Comm comm, MPI_Request *request) {
    return MPI_SUCCESS;
}
__device__ int MPI_Isend(const void *buf, int count, MPI_Datatype datatype, int dest,
                         int tag, MPI_Comm comm, MPI_Request *request) {
    return MPI_SUCCESS;
}
__device__ int MPI_Testall(int count, MPI_Request array_of_requests[],
            int *flag, MPI_Status array_of_statuses[]) {
    return MPI_SUCCESS;
}
__device__ int MPI_Waitall(int count, MPI_Request array_of_requests[],
            MPI_Status *array_of_statuses) {
    return MPI_SUCCESS;
}

__device__ int MPI_Initialized(int *flag) {
    return MPI_SUCCESS;
}

__device__ int MPI_Waitsome(int incount, MPI_Request array_of_requests[],
            int *outcount, int array_of_indices[],
            MPI_Status array_of_statuses[]) {
    return MPI_SUCCESS;
}
__device__ int MPI_Wait(MPI_Request *request, MPI_Status *status) {
    return MPI_SUCCESS;
}








