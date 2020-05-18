#ifndef COMMUNICATOR_CUH
#define COMMUNICATOR_CUH

#include "group.cuh"

struct MPI_Comm_impl;

using MPI_Comm = MPI_Comm_impl*;

__device__ extern MPI_Comm MPI_COMM_WORLD;
__device__ extern MPI_Comm MPI_COMM_NULL;

namespace gpu_mpi {

__device__ void initializeGlobalCommunicators();

__device__ int getCommContext(MPI_Comm comm);

};

__device__ int MPI_Comm_group(MPI_Comm comm, MPI_Group *group);

__device__ int MPI_Comm_create(
    MPI_Comm comm, MPI_Group group, MPI_Comm *newcomm);

__device__ int MPI_Comm_size(MPI_Comm comm, int *size);

__device__ int MPI_Comm_rank(MPI_Comm comm, int *rank);

__device__ int MPI_Cart_create(
    MPI_Comm comm_old, int ndims, const int dims[],
    const int periods[], int reorder, MPI_Comm *comm_cart);

__device__ int MPI_Cart_sub(
    MPI_Comm comm, const int remain_dims[], MPI_Comm *comm_new);

__device__ int MPI_Comm_split(
    MPI_Comm comm, int color, int key, MPI_Comm *newcomm);

__device__ int MPI_Attr_get(MPI_Comm comm, int keyval,void *attribute_val,
            int *flag );

__device__ int MPI_Attr_put(MPI_Comm comm, int keyval, void *attribute_val);

__device__ int MPI_Comm_free(MPI_Comm *comm);

#endif
