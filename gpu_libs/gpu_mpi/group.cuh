#ifndef GROUP_CUH
#define GROUP_CUH

struct MPI_Group_impl;
typedef MPI_Group_impl* MPI_Group;

__device__ extern MPI_Group MPI_GROUP_WORLD;
__device__ extern MPI_Group MPI_GROUP_NULL;
__device__ extern MPI_Group MPI_GROUP_EMPTY;

__device__ void initializeGlobalGroups();

__device__ int MPI_Group_incl(
    MPI_Group group, int n, const int ranks[], MPI_Group *newgroup);

__device__ int MPI_Group_free(MPI_Group *group);

__device__ int MPI_Group_size(MPI_Group group, int *size);

__device__ int MPI_Group_rank(MPI_Group group, int *rank);

__device__ int MPI_Group_range_incl(
    MPI_Group group, int n, int ranges[][3], MPI_Group *newgroup);

#endif
