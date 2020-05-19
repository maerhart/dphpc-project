#include "group.cuh"

#include "mpi_common.cuh"
#include "device_vector.cuh"

#include <cooperative_groups.h>
using namespace cooperative_groups;

struct MPI_Group_impl {
    CudaMPI::DeviceVector<int> ranks;
    int ref_count;
};


__device__ MPI_Group MPI_GROUP_WORLD = nullptr;
__device__ MPI_Group MPI_GROUP_EMPTY = nullptr;
__device__ MPI_Group MPI_GROUP_NULL = nullptr;

namespace gpu_mpi {

__device__ void initializeGlobalGroups() {
    if (this_grid().thread_rank() == 0) {
        MPI_GROUP_EMPTY = new MPI_Group_impl;
        
        MPI_GROUP_WORLD = new MPI_Group_impl;
        int size = this_grid().size();
        MPI_GROUP_WORLD->ranks.resize(size);
        for (int i = 0; i < size; ++i) {
            MPI_GROUP_WORLD->ranks[i] = i;
        }
    }
    this_grid().sync();
}

__device__ void destroyGlobalGroups() {
    this_grid().sync();
    if (this_grid().thread_rank() == 0) {
        delete MPI_GROUP_EMPTY;
        delete MPI_GROUP_WORLD;
    }
    
}

__device__ void incGroupRefCount(MPI_Group group) {
    if (group == MPI_GROUP_NULL || group == MPI_GROUP_WORLD || group == MPI_GROUP_EMPTY) return;
    
    assert(group->ref_count > 0);
    
    group->ref_count += 1;
}

} // namespace

__device__ int MPI_Group_incl(
    MPI_Group group, int n, const int ranks[], MPI_Group *newgroup)
{
    MPI_Group groupImpl = new MPI_Group_impl;
    groupImpl->ranks.resize(n);
    for (int i = 0; i < n; i++) {
        int rank = ranks[i];
        groupImpl->ranks[i] = group->ranks[rank];
    }
    *newgroup = groupImpl;
    return MPI_SUCCESS;
}

__device__ int MPI_Group_range_incl(
    MPI_Group group, int n, int ranges[][3], MPI_Group *newgroup)
{
    MPI_Group groupImpl = new MPI_Group_impl;
    for (int i = 0; i < n; i++) {
        int first = ranges[i][0];
        int last = ranges[i][1];
        int stride = ranges[i][2];
        if (first > last) {
            if (stride > 0) return MPI_ERROR;
            // invert range
            stride = -stride;
            int remainder = (last - first) % stride;
            int newFirst = last + remainder;
            last = first;
            first = newFirst;
        }
        
        for (int rank = first; rank <= last; rank += stride) {
            groupImpl->ranks.push_back(group->ranks[rank]);
        }
    }
    *newgroup = groupImpl;
    return MPI_SUCCESS;
}

__device__ int MPI_Group_free(MPI_Group *group) {
    if ((*group) == MPI_GROUP_NULL || (*group) == MPI_GROUP_WORLD || (*group) == MPI_GROUP_EMPTY) {
        (*group) = MPI_GROUP_NULL;
        return MPI_SUCCESS;
    }
    
    assert((*group)->ref_count > 0);
    (*group)->ref_count--;
    if ((*group)->ref_count == 0) delete *group;
    *group = MPI_GROUP_NULL;
    return MPI_SUCCESS;
}

__device__ int MPI_Group_size(MPI_Group group, int *size) {
    *size = group->ranks.size();
    return MPI_SUCCESS;
}

__device__ int MPI_Group_rank(MPI_Group group, int *rank) {
    int currentRank = this_multi_grid().thread_rank();
    
    for (int groupRank = 0; groupRank < group->ranks.size(); groupRank++) {
        int globalRank = group->ranks[groupRank];
        if (currentRank == globalRank) {
            *rank = groupRank;
            return MPI_SUCCESS;
        }
    }
    
    return MPI_UNDEFINED;
}


