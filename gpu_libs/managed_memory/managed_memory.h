#ifndef MANAGED_MEMORY_H
#define MANAGED_MEMORY_H

#include <cuda.h>

class ManagedMemory
{
public:
    __host__ __device__ void lock() {
#if defined(__CUDA_ARCH__)
        while (atomicCAS_system(&locked, 0, 1)) {}
#else
        while (__sync_val_compare_and_swap(&locked, 0, 1)) {}
#endif
    }

    __host__ __device__ void unlock() {
#if defined(__CUDA_ARCH__)
        __threadfence_system(); // this is really required, prevents race condition
#endif
        locked = 0;
    }

    // 64 Mb
    enum { MEMORY_SIZE = 64 * (1 << 20) };

    // volatile is really required, prevents reading cached values
    volatile char memory[MEMORY_SIZE];
private:
    unsigned locked;
};

class HostDeviceManagedMemory
{
public:
    __host__ __device__ void commitChange() {
        #if defined(__CUDA_ARCH__)
            __threadfence_system(); // this is really required, prevents race condition
            ownedByHost = true;
        #else
            __sync_synchronize();
            ownedByHost = false;
        #endif
    }

    __host__ __device__ bool isChanged() {
        #if defined(__CUDA_ARCH__)
            return !ownedByHost;
        #else
            return ownedByHost;
        #endif
    }

    // 64 Mb
    enum { MEMORY_SIZE = 64 * (1 << 20) };

    // volatile is really required, prevents reading cached values
    volatile char memory[MEMORY_SIZE];
    bool ownedByHost;
};

extern __managed__ ManagedMemory gManagedMemory;

#endif // MANAGED_MEMORY_H
