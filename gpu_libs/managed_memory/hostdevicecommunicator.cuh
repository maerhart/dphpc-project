#ifndef HOSTMESSAGEQUEUE_H
#define HOSTMESSAGEQUEUE_H

struct HostDeviceMessage {
    enum MessageType {
        FPRINTF
    };

    void process();
    // 1 Mb per thread
    enum { MEMORY_PER_THREAD = 1 << 20 };

    volatile unsigned mIsOwnedByHost;
    volatile int mMessageType; 
    volatile char mMemory[MEMORY_PER_THREAD];
};

struct HostDeviceCommunicatior {
    void init(int gridSize, int blockSize);
    void destroy();

    __device__ void putMessage();
    
    void processMessages();

    __device__ HostDeviceMessage* currentThreadMemoryPosition();

    int mGridSize;
    int mBlockSize;

    int mManagedMemorySize;
    HostDeviceMessage* mManagedMemory;
};

extern __managed__ HostDeviceCommunicatior gHostDeviceCommunicator;

#endif // HOSTMESSAGEQUEUE_H
