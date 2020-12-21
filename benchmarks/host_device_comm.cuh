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

template <typename T>
__forceinline__
__host__ __device__
volatile T& volatileAccess(T& val) {
    volatile T* vval = &val;
    return *vval;
}

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

template <typename T>
class ManagedVector {
public:
    template <typename... Args>
    __host__ __device__ ManagedVector(int size, Args... args) : mSize(size)
    {
#ifdef __CUDA_ARCH__
        printf("ERROR: ManagedVector can't be initialized from device");
        assert(0);
#else
        CUDA_CHECK(cudaMallocManaged(&mData, mSize * sizeof(T)));
        assert(mData);
        for (int i = 0; i < size; i++) {
            new (&mData[i]) T(args...);
        }
#endif
    }

    __host__ __device__ ~ManagedVector() {
#ifdef __CUDA_ARCH__
        assert(0);
#else
        for (int i = 0; i < mSize; i++) {
            mData[i].~T();
        }
        CUDA_CHECK(cudaFree((T*)mData));
#endif
    }
    
    __host__ __device__ volatile T& operator[](int index) volatile {
        assert(0 <= index && index < mSize);
        return VOLATILE(mData[index]);
    }

    __host__ __device__ int size() const volatile { return mSize; }

private:
    int mSize;
    T* mData;
};

struct CircularBufferState {
    __host__ __device__ CircularBufferState(int size)
        : size(size)
        , used(0)
        , head(0)
        , tail(0)
    {
    }

    __host__ __device__ bool empty() const volatile { return used == 0; }
    __host__ __device__ bool full() const volatile { return used == size; }

    // reserve and return position for new element at the tail of queue
    __host__ __device__ int push() volatile {
        assert(!full());
        used += 1;
        int position = tail;
        tail = (position + 1) % size;
        return position;
    }

    // release and return (released) position of element from the head of queue
    __host__ __device__ int pop() volatile {
        assert(!empty());
        used -= 1;
        int position = head;
        head = (position + 1) % size;
        return position;
    }

    int used;
    int head; // first
    int tail; // next after last
    int size;
};

template <typename T, template <typename> typename Vector>
class CircularQueue {
public:
    
    __host__ __device__ CircularQueue(int size)
        : data(size)
        , active(size, false)
        , bufferState(size)
    {
    }
    
    __host__ __device__ int size() volatile {
        return bufferState.size;
    }
    
    __host__ __device__ int used() volatile {
        return bufferState.used;
    }
    
    __host__ __device__ bool empty() volatile {
        return bufferState.empty();
    }
    
    __host__ __device__ int full() volatile {
        return bufferState.full();
    }

    __host__ __device__ int push(const T& md) volatile {
        int position = bufferState.push();
        data[position] = md;
        active[position] = true;
        return position;
    }
    
    __host__ __device__ T& get(int position) volatile {
        assert(0 <= position && position < data.size());
        assert(active[position]);
        return data[position];
    }
    
    __host__ __device__ void pop(volatile T* elem) volatile {
        int index = elem - &data[0];
        assert(0 <= index && index < data.size());
        active[index] = false;
        if (bufferState.head == index) {
            while (!active[index] && !bufferState.empty()) {
                int removedIndex = bufferState.pop();
                assert(removedIndex == index);
                index = (index + 1) % data.size();
            }
        }
    }

    __host__ __device__ volatile T* head() volatile {
        if (bufferState.empty()) return nullptr;
        assert(active[bufferState.head]);
        return &data[bufferState.head];
    }

    __host__ __device__ volatile T* next(volatile T* elem) volatile {
        assert(elem);
        int curIndex = elem - &data[0];
        auto next = [size=data.size()] (int idx) { return (idx + 1) % size; };
        for (int idx = next(curIndex); idx != bufferState.tail; idx = next(idx)) {
            if (active[idx]) return &data[idx];
        }
        return nullptr;
    }

private:
    Vector<T> data;
    Vector<bool> active;
    CircularBufferState bufferState;
};

template <typename T>
using ManagedCircularQueue = CircularQueue<T, ManagedVector>;

/*
 * Code of this function is taken from cooperative_groups::details::sync_grids()
 * defined in cooperative_groups/details/sync.h
 */
__forceinline__ __device__
void myCudaBarrier(unsigned int expected, volatile unsigned int* arrived, bool master) {
    bool cta_master = (threadIdx.x + threadIdx.y + threadIdx.z == 0);

    __syncthreads();

    if (cta_master) {
        unsigned int nb = 1;
        if (master) {
            nb = 0x80000000 - (expected - 1);
        }

        __threadfence();

        unsigned int oldArrive;
        oldArrive = cooperative_groups::details::atomic_add(arrived, nb);

        while (!cooperative_groups::details::bar_has_flipped(oldArrive, *arrived));

        //flush barrier upon leaving
        cooperative_groups::details::bar_flush((unsigned int*)arrived);
    }

    __syncthreads();
}
