#include "mpi.h.cuh"

__device__ int MPI_Init(int *argc, char ***argv) {
    return MPI_SUCCESS;
}

__device__ int MPI_Finalize(void) {
    return MPI_SUCCESS;
}

__device__ int MPI_Comm_size(MPI_Comm comm, int *size) {
    *size = 1;
    return MPI_SUCCESS;
}

__device__ int MPI_Comm_rank(MPI_Comm comm, int *rank) {
    *rank = 0;
    return MPI_SUCCESS;
}

__device__ int MPI_Get_processor_name(char *name, int *resultlen) {
    return MPI_SUCCESS;
}

__device__ int MPI_Bcast(void *buffer, int count, MPI_Datatype datatype,
                         int root, MPI_Comm comm)
{
    return MPI_SUCCESS;
}

__device__ double MPI_Wtime(void) {
    return 0.;
}

__device__ int MPI_Reduce(const void *sendbuf, void *recvbuf, int count,
                          MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm)
{
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



__device__ MPI_Comm MPI_COMM_WORLD;

__device__ MPI_Datatype MPI_INT;
__device__ MPI_Datatype MPI_DOUBLE;
__device__ MPI_Datatype MPI_CHAR;

__device__ MPI_Op MPI_SUM;
