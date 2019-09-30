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

__device__ double MPI_Wtick() {
    return 0.0;
}
__device__ int MPI_Comm_group(MPI_Comm comm, MPI_Group *group) {
    return MPI_SUCCESS;
}
__device__ int MPI_Group_incl(MPI_Group group, int n, const int ranks[],
            MPI_Group *newgroup) {
    return MPI_SUCCESS;
}
__device__ int MPI_Comm_create(MPI_Comm comm, MPI_Group group, MPI_Comm *newcomm) {
    return MPI_SUCCESS;
}
__device__ int MPI_Group_free(MPI_Group *group) {
    return MPI_SUCCESS;
}
__device__ int MPI_Group_rank(MPI_Group group, int *rank) {
    return MPI_SUCCESS;
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
__device__ int MPI_Attr_get(MPI_Comm comm, int keyval,void *attribute_val,
            int *flag ) {
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
             MPI_Datatype recvtype, MPI_Comm comm) {
    return MPI_SUCCESS;
}
__device__ int MPI_Allgatherv(const void *sendbuf, int sendcount,
                              MPI_Datatype sendtype, void *recvbuf, const int recvcounts[],
                              const int displs[], MPI_Datatype recvtype, MPI_Comm comm) {
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
__device__ int MPI_Attr_put(MPI_Comm comm, int keyval, void *attribute_val) {
    return MPI_SUCCESS;
}
__device__ int MPI_Dims_create(int nnodes, int ndims, int dims[]) {
    return MPI_SUCCESS;
}
__device__ int MPI_Cart_create(MPI_Comm comm_old, int ndims, const int dims[],
                               const int periods[], int reorder, MPI_Comm *comm_cart) {
    return MPI_SUCCESS;
}
__device__ int MPI_Cart_sub(MPI_Comm comm, const int remain_dims[], MPI_Comm *comm_new) {
    return MPI_SUCCESS;
}
__device__ int MPI_Comm_split(MPI_Comm comm, int color, int key,
                              MPI_Comm *newcomm) {
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
__device__ int MPI_Group_range_incl(MPI_Group group, int n, int ranges[][3],
                                    MPI_Group *newgroup) {
    return MPI_SUCCESS;
}
__device__ int MPI_Comm_free(MPI_Comm *comm) {
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






__device__ MPI_Comm MPI_COMM_WORLD;


