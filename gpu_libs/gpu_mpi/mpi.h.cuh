#ifndef MPI_H
#define MPI_H


#define MPI_MAX_PROCESSOR_NAME 256

#define MPI_ANY_TAG -1

#include "mpi_common.cuh"
#include "group.cuh"
#include "communicator.cuh"

typedef int MPI_Datatype;
typedef int MPI_Op;
typedef struct MPI_Status_t {} MPI_Status;
typedef struct MPI_Request_t {} MPI_Request;

__device__ int MPI_Init(int *argc, char ***argv);
__device__ int MPI_Finalize(void);

__device__ int MPI_Get_processor_name(char *name, int *resultlen);
__device__ int MPI_Bcast(void *buffer, int count, MPI_Datatype datatype,
                     int root, MPI_Comm comm);
__device__ double MPI_Wtime(void);
__device__ int MPI_Reduce(const void *sendbuf, void *recvbuf, int count,
                      MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm);

__device__ int MPI_Type_contiguous(int count, MPI_Datatype oldtype, MPI_Datatype *newtype);
__device__ int MPI_Type_commit(MPI_Datatype *datatype);
__device__ int MPI_Recv(void *buf, int count, MPI_Datatype datatype,
            int source, int tag, MPI_Comm comm, MPI_Status *status);
__device__ int MPI_Sendrecv(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
            int dest, int sendtag, void *recvbuf, int recvcount,
            MPI_Datatype recvtype, int source, int recvtag,
                 MPI_Comm comm, MPI_Status *status);
__device__ int MPI_Send(const void *buf, int count, MPI_Datatype datatype, int dest,
            int tag, MPI_Comm comm);
__device__ double MPI_Wtick();
__device__ int MPI_Allreduce(const void *sendbuf, void *recvbuf, int count,
                         MPI_Datatype datatype, MPI_Op op, MPI_Comm comm);
__device__ int MPI_Abort(MPI_Comm comm, int errorcode);
__device__ int MPI_Type_size(MPI_Datatype datatype, int *size);
__device__ int MPI_Gather(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
            void *recvbuf, int recvcount, MPI_Datatype recvtype, int root,
            MPI_Comm comm);

__device__ int MPI_Barrier(MPI_Comm comm);
__device__ int MPI_Alltoall(const void *sendbuf, int sendcount,
            MPI_Datatype sendtype, void *recvbuf, int recvcount,
            MPI_Datatype recvtype, MPI_Comm comm);
__device__ int MPI_Alltoallv(const void *sendbuf, const int sendcounts[],
            const int sdispls[], MPI_Datatype sendtype,
            void *recvbuf, const int recvcounts[],
            const int rdispls[], MPI_Datatype recvtype, MPI_Comm comm);
__device__ int MPI_Allgather(const void *sendbuf, int  sendcount,
             MPI_Datatype sendtype, void *recvbuf, int recvcount,
             MPI_Datatype recvtype, MPI_Comm comm);
__device__ int MPI_Allgatherv(const void *sendbuf, int sendcount,
                              MPI_Datatype sendtype, void *recvbuf, const int recvcounts[],
                              const int displs[], MPI_Datatype recvtype, MPI_Comm comm);
__device__ int MPI_Gatherv(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                           void *recvbuf, const int recvcounts[], const int displs[], MPI_Datatype recvtype,
                           int root, MPI_Comm comm);
__device__ int MPI_Scatter(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                           void *recvbuf, int recvcount, MPI_Datatype recvtype, int root,
                           MPI_Comm comm);

typedef int MPI_Copy_function(MPI_Comm oldcomm, int keyval,
                                       void *extra_state, void *attribute_val_in,
                                       void *attribute_val_out, int *flag);
__device__ int MPI_NULL_COPY_FN(MPI_Comm oldcomm, int keyval,
                     void *extra_state, void *attribute_val_in,
                     void *attribute_val_out, int *flag);

typedef int MPI_Delete_function(MPI_Comm comm, int keyval,
             void *attribute_val, void *extra_state);
__device__ int MPI_NULL_DELETE_FN(MPI_Comm comm, int keyval,
                       void *attribute_val, void *extra_state);


__device__ int MPI_Keyval_create(MPI_Copy_function *copy_fn,
                                 MPI_Delete_function *delete_fn, int *keyval, void *extra_state);
__device__ int MPI_Dims_create(int nnodes, int ndims, int dims[]);

__device__ int MPI_Irecv(void *buf, int count, MPI_Datatype datatype,
               int source, int tag, MPI_Comm comm, MPI_Request *request);
__device__ int MPI_Isend(const void *buf, int count, MPI_Datatype datatype, int dest,
                         int tag, MPI_Comm comm, MPI_Request *request);
__device__ int MPI_Testall(int count, MPI_Request array_of_requests[],
            int *flag, MPI_Status array_of_statuses[]);
__device__ int MPI_Waitall(int count, MPI_Request array_of_requests[],
            MPI_Status *array_of_statuses);
__device__ int MPI_Initialized(int *flag);

__device__ int MPI_Waitsome(int incount, MPI_Request array_of_requests[],
            int *outcount, int array_of_indices[],
            MPI_Status array_of_statuses[]);
__device__ int MPI_Wait(MPI_Request *request, MPI_Status *status);

#define MPI_INT 0
#define MPI_DOUBLE 1
#define MPI_CHAR 2
#define MPI_BYTE 3
#define MPI_DOUBLE_INT 4
#define MPI_LONG_LONG 5

#define MPI_SUM 1
#define MPI_MIN 2
#define MPI_MAX 3
#define MPI_MAXLOC 4

#define MPI_STATUSES_IGNORE ((MPI_Status*)1)
#define MPI_STATUS_IGNORE ((MPI_Status*)1)

#endif // MPI_H
