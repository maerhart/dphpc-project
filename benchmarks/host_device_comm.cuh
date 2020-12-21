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
    return val;
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
        mData = (T*) malloc(mSize * sizeof(T));
#else
        CUDA_CHECK(cudaMallocManaged(&mData, mSize * sizeof(T)));
        
#endif
        assert(mData);
        for (int i = 0; i < size; i++) {
            new (&mData[i]) T(args...);
        }
    }

    __host__ __device__ ~ManagedVector() {
        for (int i = 0; i < mSize; i++) {
            mData[i].~T();
        }
#ifdef __CUDA_ARCH__
        free(mData);
#else
        CUDA_CHECK(cudaFree(mData));
#endif
    }
    
    __host__ __device__ T& operator[](int index) {
        if (!(0 <= index && index < mSize)) {
            printf("Index out of range\n");
        }
        assert(0 <= index && index < mSize);
        return mData[index];
    }

    __host__ __device__ int size() const { return mSize; }

private:
    int mSize;
    T* mData;
};



struct QueueState {
    using Int32 = int;
    using UInt64 = unsigned long long;
    
    __host__ __device__ QueueState(int size)
        : size(size)
        , state(0)
    {
    }

    struct Impl {
        static __device__ bool empty(const UInt64& state) { return used(state) == 0; }
        static __device__ bool full(const UInt64& state, int size) { return used(state) == size; }

        static __device__ const Int32& head(const UInt64& state) {
            return ((const int*)&state)[0];
        }
        
        static __device__ const Int32& used(const UInt64& state) {
            return ((const int*)&state)[1];
        }
        
        static __device__ Int32& head(UInt64& state) {
            return ((int*)&state)[0];
        }
        
        static __device__ Int32& used(UInt64& state) {
            return ((int*)&state)[1];
        }
    };
    
    __device__ bool empty() const { 
        UInt64 s = volatileAccess(state);
        return Impl::empty(s);
    }
    __device__ bool full() const { 
        UInt64 s = volatileAccess(state);
        return Impl::full(s, size);
    }
    __device__ Int32& head() {
        UInt64 s = volatileAccess(state);
        return Impl::head(s);
    }
    __device__ Int32& used() {
        UInt64 s = volatileAccess(state);
        return Impl::used(s);
    }
    
    // reserve and return position for new element at the tail of queue
    __device__ bool try_push(int& pos) {
        UInt64 oldState = volatileAccess(state);
        if (Impl::full(oldState, size)) return false;
        UInt64 newState;
        Impl::head(newState) = Impl::head(oldState);
        Impl::used(newState) = Impl::used(oldState) + 1;
        if (oldState == atomicCAS(&state, oldState, newState)) {
            // success, old state is not changed
            pos = (Impl::head(oldState) + Impl::used(oldState)) % size;
            //printf("Try push s, state = (size = %d, used = %d)\n", Impl::head(state), Impl::used(state));
            return true;
        }
        return false;
    }

    // release and return (released) position of element from the head of queue
    __device__ bool try_pop(int& pos) {
        //printf("Try pop 1, state = (size = %d, used = %d)\n", Impl::head(state), Impl::used(state));
        UInt64 oldState = volatileAccess(state);
        if (Impl::empty(oldState)) {
            //printf("Try pop f1, state = (size = %d, used = %d)\n", Impl::head(state), Impl::used(state));
            return false;
        }
        UInt64 newState;
        Impl::head(newState) = (Impl::head(oldState) + 1) % size;
        Impl::used(newState) = Impl::used(oldState) - 1;
        if (oldState == atomicCAS(&state, oldState, newState)) {
            // success, old state is not changed
            pos = Impl::head(oldState);
            //printf("Try pop s, state = (size = %d, used = %d)\n", Impl::head(state), Impl::used(state));
            return true;
        }
        //printf("Try pop f2, state = (size = %d, used = %d)\n", Impl::head(state), Impl::used(state));
        return false;
    }
    
    __device__ int push() {
        int pos = -1;
        while (!try_push(pos)) {}
//         printf("Pushed, state = (size = %d, used = %d)\n", Impl::head(state), Impl::used(state));
        return pos;
    }
    
    __device__ int pop() {
        int pos = -1;
        while (!try_pop(pos)) {}
//         printf("Popped, state = (size = %d, used = %d)\n", Impl::head(state), Impl::used(state));
        return pos;
    }
    
    __device__ int top() {
        UInt64 s = 0;
        do {
            s = volatileAccess(state);
        } while (Impl::empty(s));
        return Impl::head(s);
    }
    
    int size;
private:
    UInt64 state;
};

template <typename T>
class Queue {
public:
    
    struct Elem {
        bool valid = false;
        T value;
    };
    
    __host__ __device__ Queue(int size)
        : data(size)
        , queueState(size)
    {
    }
    
    __device__ int size() {
        return queueState.size;
    }
    
    __device__ int used() {
        return queueState.used();
    }
    
    __device__ bool empty() {
        return queueState.empty();
    }
    
    __device__ int full() {
        bool res = queueState.full();
        if (res) printf("The queue is full\n");
        return res;
    }

    __device__ void push(const T& val) {
        while (!try_push(val)) {}
    }
    
    __device__ T pop() {
        T val;
        while (!try_pop(val)) {}
        return val;
    }

    __device__ bool try_push(const T& val) {
        int pos = -1;
        if (queueState.try_push(pos)) {
            WAIT(data[pos].valid == false); // TODO: rewrite without WAIT
            volatileAccess(data[pos].value) = val;
            memfence();
            volatileAccess(data[pos].valid) = true;
            return true;
        }
        return false;
    }
    
    __device__ bool try_pop(T& val) {
        int pos = -1;
        if (queueState.try_pop(pos)) {
            WAIT(data[pos].valid == true); // TODO: rewrite without WAIT
            val = volatileAccess(data[pos].value);
            memfence();
            volatileAccess(data[pos].valid) = false;
            return true;
        }
        return false;
    }
        
private:
    ManagedVector<Elem> data;
    QueueState queueState;
};

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
