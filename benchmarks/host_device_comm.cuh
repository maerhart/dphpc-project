#pragma once

#include <cuda.h>

// internal CUDA header
// copy paste this headers from cuda 11.1 if at some point they break
#include <cooperative_groups/details/sync.h>
#include <cooperative_groups/details/helpers.h>

#define CUDA_CHECK(expr) \
    do { if ((expr) != cudaSuccess) { \
        printf("CUDA_ERROR %s:%d %s\n", __FILE__, __LINE__, #expr); \
        abort(); \
    } } while (0)

__forceinline__ __host__ __device__ void memfence() {
#if __CUDA_ARCH__
    __threadfence_system();
#else
    __sync_synchronize();
#endif
}

__forceinline__ __device__ uint64_t globaltime()
{
    uint64_t res;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(res) );
    return res;
}

#define WAIT(condition) do { memfence(); while(!(condition)) {} } while(0)

class HostDeviceComm {
public:
    HostDeviceComm()
        : deviceData(nullptr)
        , hostData(nullptr)
    {
        CUDA_CHECK(cudaMalloc(&deviceData, sizeof(*deviceData)));
        CUDA_CHECK(cudaMemset(deviceData, 0, sizeof(*deviceData)));

        CUDA_CHECK(cudaMallocHost(&hostData, sizeof(*hostData)));
        CUDA_CHECK(cudaMemset(hostData, 0, sizeof(*hostData)));
    }

    ~HostDeviceComm() {
        CUDA_CHECK(cudaFree(deviceData));
        CUDA_CHECK(cudaFreeHost(hostData));
    }

    __device__ unsigned long long rank() {
        return cooperative_groups::details::grid::thread_rank();
    }

    __device__ unsigned long long size() {
        return cooperative_groups::details::grid::size();
    }

    __device__ void deviceBarrier() {
        cooperative_groups::details::sync_grids(size(), &deviceData->arrived);
    }

    __host__ __device__ void hostDeviceBarrier() {
#ifdef __CUDA_ARCH__
        if (rank() == 0) syncWithHostFromDevice();
        deviceBarrier();
        if (rank() == 0) syncWithHostFromDevice();
#else
        // twice, it is not a bug!
        syncWithDeviceFromHost();
        syncWithDeviceFromHost();
#endif
    }

private:
    __device__ void syncWithHostFromDevice() {
        WAIT(hostData->deviceBarrierReady == false);
        hostData->deviceBarrierReady = true;
        WAIT(hostData->hostBarrierReady == true);
        hostData->hostBarrierReady = false;
    }

    void syncWithDeviceFromHost() {
        WAIT(hostData->hostBarrierReady == false);
        hostData->hostBarrierReady = true;
        WAIT(hostData->deviceBarrierReady == true);
        hostData->deviceBarrierReady = false;
    }
  
    struct DeviceData {
        volatile unsigned int arrived;
    };
    DeviceData* deviceData;

    struct HostData {
        volatile bool hostBarrierReady;
        volatile bool deviceBarrierReady;
    };
    HostData* hostData;
};

