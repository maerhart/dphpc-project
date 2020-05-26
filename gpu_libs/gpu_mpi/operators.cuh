#pragma once

#include "datatypes.cuh"

#include "mpi_operators_list.cuh"

struct MPI_Op_impl;
using MPI_Op = MPI_Op_impl*;

#define MPI_OP_LIST_DECL_F(name, class) __device__ extern MPI_Op name;
#define MPI_OP_LIST_DECL_SEP

MPI_OPERATORS_LIST(MPI_OP_LIST_DECL_F, MPI_OP_LIST_DECL_SEP)

#undef MPI_OP_LIST_DECL_F
#undef MPI_OP_LIST_DECL_SEP

typedef void MPI_User_function(void* invec, void* inoutvec, int* len, MPI_Datatype* datatype);

#define MPI_OP_NULL nullptr

namespace gpu_mpi {

__device__ void initializeOps();
__device__ void destroyOps();
__device__ void invokeOperator(MPI_Op op, const void* in, void* inout, int* len, MPI_Datatype* datatype);

} // namespace
