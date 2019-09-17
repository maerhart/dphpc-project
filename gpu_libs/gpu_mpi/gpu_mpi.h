#ifndef MPI_H
#define MPI_H

#define MPI_SUCCESS 0

#define MPI_MAX_PROCESSOR_NAME 256

typedef struct MPI_Comm_t {} MPI_Comm;
typedef struct MPI_Datatype_t {} MPI_Datatype;
typedef struct MPI_Op_t {} MPI_Op;

__device__ int MPI_Init(int *argc, char ***argv);
__device__ int MPI_Finalize(void);
__device__ int MPI_Comm_size(MPI_Comm comm, int *size);
__device__ int MPI_Comm_rank(MPI_Comm comm, int *rank);
__device__ int MPI_Get_processor_name(char *name, int *resultlen);
__device__ int MPI_Bcast(void *buffer, int count, MPI_Datatype datatype,
                     int root, MPI_Comm comm);
__device__ double MPI_Wtime(void);
__device__ int MPI_Reduce(const void *sendbuf, void *recvbuf, int count,
                      MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm);

__device__ extern MPI_Comm MPI_COMM_WORLD;

__device__ extern MPI_Datatype MPI_INT;
__device__ extern MPI_Datatype MPI_DOUBLE;

__device__ extern MPI_Op MPI_SUM;

#endif // MPI_H
