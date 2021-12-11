#pragma once


#include "mpi.cuh"


#ifdef GPUMPI_MALLOC_COALESCE
#define MPI_Alltoall MPI_Alltoall_coalesce
#define MPI_Alltoallv MPI_Alltoallv_coalesce
#endif