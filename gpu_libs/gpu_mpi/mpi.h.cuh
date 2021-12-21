#pragma once


#include "mpi.cuh"

#ifdef GPUMPI_MALLOC_V1
#define MPI_Alltoall MPI_Alltoall_v1
#define MPI_Alltoallv MPI_Alltoallv_v1
#endif
#ifdef GPUMPI_MALLOC_V2
#define MPI_Alltoall MPI_Alltoall_v2
#define MPI_Alltoallv MPI_Alltoallv_v2
#endif
#ifdef GPUMPI_MALLOC_V3
#define MPI_Alltoall MPI_Alltoall_v3
#define MPI_Alltoallv MPI_Alltoallv_v3
#endif
#ifdef GPUMPI_MALLOC_V4
#define MPI_Alltoall MPI_Alltoall_v4
#define MPI_Alltoallv MPI_Alltoallv_v4
#endif
#ifdef GPUMPI_MALLOC_V5
#define MPI_Alltoall MPI_Alltoall_v5
#define MPI_Alltoallv MPI_Alltoallv_v5
#endif
#ifdef GPUMPI_MALLOC_V6
#define MPI_Alltoall MPI_Alltoall_coalesce
#define MPI_Alltoallv MPI_Alltoallv_coalesce
#endif
#ifdef GPUMPI_MALLOC_V7
#define MPI_Alltoall MPI_Alltoall_v6
#define MPI_Alltoallv MPI_Alltoallv_v6
#endif
