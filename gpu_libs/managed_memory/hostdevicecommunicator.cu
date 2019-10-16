#include "hostdevicecommunicator.cuh"

#include "common.h"
#include "assert.h.cuh"

__managed__ HostDeviceCommunicatior gHostDeviceCommunicator;

void HostDeviceCommunicatior::init(int gridSize, int blockSize)
{
    mGridSize = gridSize;
    mBlockSize = blockSize;
    mManagedMemorySize = gridSize * blockSize * sizeof(*mManagedMemory);
    mManagedMemory = nullptr;
    CUDA_CHECK(cudaMallocManaged((void**) &mManagedMemory, mManagedMemorySize));
    memset(mManagedMemory, 0, mManagedMemorySize);
}

void HostDeviceCommunicatior::destroy()
{
    CUDA_CHECK(cudaFree(mManagedMemory));
}

__device__ HostDeviceMessage* HostDeviceCommunicatior::currentThreadMemoryPosition() {
    assert(gridDim.x == mGridSize);
    assert(blockDim.x == mBlockSize);
    int currentThreadIdx = threadIdx.x + blockDim.x * blockIdx.x;
    assert(0 <= currentThreadIdx && currentThreadIdx < mGridSize * mBlockSize);
    return mManagedMemory + currentThreadIdx;
}

void HostDeviceCommunicatior::processMessages() {
    for (int threadIdx = 0; threadIdx < mGridSize * mBlockSize; threadIdx++) {
        HostDeviceMessage& hostDeviceMessage = mManagedMemory[threadIdx];
        if (hostDeviceMessage.mIsOwnedByHost == 0) {
            hostDeviceMessage.process();
            
        }
    }
}

void HostDeviceMessage::process() {
    //hostDeviceMessage
}
