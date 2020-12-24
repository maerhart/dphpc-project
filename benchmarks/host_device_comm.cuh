#pragma once

#include <cuda.h>

// internal CUDA header
// copy paste this headers from cuda 11.1 if at some point they break
#include <cooperative_groups/details/sync.h>
#include <cooperative_groups/details/helpers.h>

#define CUDA_CHECK(expr) \
    do { \
        cudaError_t err = (expr); \
        if (err != cudaSuccess) { \
            printf("CUDA_ERROR %s:%d %s %s\n", __FILE__, __LINE__, #expr, cudaGetErrorString(err)); \
            abort(); \
        } \
    } while (0)

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

#define DEBUG_WAIT(condition) do {\
    int counter = 0;\
    memfence();\
    while(!(condition)) {\
        if (counter < 0) continue;\
        counter++;\
        if (counter >= 100000000) {\
            counter = -1;\
            printf("Potential livelock at %s:%d threadIdx.x = %d blockIdx.x = %d\n", __FILE__, __LINE__, threadIdx.x, blockIdx.x);\
        }\
    }\
    if (counter < 0) {\
        printf("No livelock at %s:%d threadIdx.x = %d blockIdx.x = %d\n", __FILE__, __LINE__, threadIdx.x, blockIdx.x);\
    }\
} while(0)

#define PLAIN_WAIT(condition) do {\
    memfence();\
    while(!(condition)) {}\
} while(0)

#define SLEEP_WAIT(condition) do {\
    memfence();\
    long long _wait_start = clock64();\
    long long _sleep_time = 1;\
    while(true) {\
        if (condition) break;\
        else {\
            if (_sleep_time < 1000) {\
                _sleep_time *= 2;\
            }\
            long long _current_clock = clock64();\
            while (clock64() - _current_clock < _sleep_time) {}\
        }\
    }\
} while(0)

#define WAIT DEBUG_WAIT

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
        PLAIN_WAIT(hostData->deviceBarrierReady == false);
        hostData->deviceBarrierReady = true;
        PLAIN_WAIT(hostData->hostBarrierReady == true);
        hostData->hostBarrierReady = false;
    }

    void syncWithDeviceFromHost() {
        PLAIN_WAIT(hostData->hostBarrierReady == false);
        hostData->hostBarrierReady = true;
        PLAIN_WAIT(hostData->deviceBarrierReady == true);
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
        //assert(0 <= index && index < mSize);
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

// template <typename T>
// class Queue {
// public:
//     using Int16 = short;
//     
//     __host__ __device__ Queue(Int16 size)
//         : data(size)
//     {
//     }
//     
//     __device__ Int16 size() {
//         return data.size();
//     }
// 
//     __device__ bool reserveElem(Int16& reservedIndex) {
//         State oldState = state;
//         if (oldState.part.reserved == data.size()) {
//             queue is full
//             return false;
//         }
//         State newState = oldState;
//         newState.part.reserved += 1;
//         if (oldState.full == atomicCAS(&state.full, oldState.full, newState.full)) {
//             success, old state is not changed
//             reservedIndex = (oldState.part.head + oldState.part.reserved) % size();
//             return true;
//         }
//         return false;
//     }
//     
//     __device__ bool makeElemValid(Int16 elemIndex) {
//         State oldState = state;
//         if ((oldState.part.head + oldState.part.valid) % size() != elemIndex) {
//             not all elements before elemIndex are valid, so can't validate current elem as well
//             return false;
//         }
//         State newState = oldState;
//         newState.part.valid += 1;
//         if (oldState.full == atomicCAS(&state.full, oldState.full, newState.full)) {
//             success, old state is not changed
//             return true;
//         }
//         return false;
//     }
//     
//     __device__ bool tryPop(T& elem) {
//         State oldState = state;
//         if (oldState.part.valid == 0) {
//             queue is empty
//             return false;
//         }
//         State newState = oldState;
//         newState.part.valid -= 1;
//         newState.part.reserved -= 1;
//         newState.part.head = (oldState.part.head + 1) % size();
//         T potentialElem = data[oldState.part.head];
//         memfence(); // pop element only after reading the value
//         if (oldState.full == atomicCAS(&state.full, oldState.full, newState.full)) {
//             success, old state is not changed
//             elem = potentialElem;
//             printf("Pop value %d\n", elem);
//             return true;
//         }
//         return false;
//     }
//     
//     __device__ void push(const T& val) {
//         Int16 reservedIndex = -1;
//         WAIT(reserveElem(reservedIndex));
//         data[reservedIndex] = val;
//         memfence(); // make elem valid only after assigning it
//         WAIT(makeElemValid(reservedIndex));
//         printf("Push value %d\n", val);
//     }
//     
//     __device__ T pop() {
//         T elem;
//         WAIT(tryPop(elem));
//         return elem;
//     }
//         
// private:
//     using UInt64 = unsigned long long;
//     
//     union State {
//         __host__ __device__ State() : full(0) {}
//         struct {
//             Int16 head; // index of head
//             Int16 valid; // number of valid elements
//             Int16 reserved; // number of reserved elements (reserved <= valid <= reserved + 1)
//         } part;
//         UInt64 full;
//     } state;
// 
//     static_assert(sizeof(Int16) == 2);
//     static_assert(sizeof(UInt64) == 8);
//     static_assert(sizeof(State) == 8);
//     
//     ManagedVector<T> data;
// };

template <typename T>
class Queue {
public:
    using Int16 = short;
    
    __host__ __device__ Queue(Int16 size)
        : data(size)
    {
    }
    
    __device__ Int16 size() {
        return data.size();
    }

    __device__ bool reserveElem(Int16& reservedIndex) {
        State oldState = state;
        if (oldState.part.reserved == data.size()) {
            // queue is full
            return false;
        }
        State newState = oldState;
        newState.part.reserved += 1;
        if (oldState.full == atomicCAS(&state.full, oldState.full, newState.full)) {
            // success, old state is not changed
            reservedIndex = (oldState.part.head + oldState.part.reserved) % size();
            return true;
        }
        return false;
    }
    
    __device__ bool makeElemValid(Int16 elemIndex) {
        State oldState = state;
        if ((oldState.part.head + oldState.part.valid) % size() != elemIndex) {
            // not all elements before elemIndex are valid, so can't validate current elem as well
            return false;
        }
        State newState = oldState;
        newState.part.valid += 1;
        if (oldState.full == atomicCAS(&state.full, oldState.full, newState.full)) {
            // success, old state is not changed
            return true;
        }
        return false;
    }
    
    __device__ bool tryPop(T& elem) {
        State oldState = state;
        if (oldState.part.valid == 0) {
            // queue is empty
            return false;
        }
        State newState = oldState;
        newState.part.valid -= 1;
        newState.part.reserved -= 1;
        newState.part.head = (oldState.part.head + 1) % size();
        T potentialElem = data[oldState.part.head];
        memfence(); // pop element only after reading the value
        if (oldState.full == atomicCAS(&state.full, oldState.full, newState.full)) {
            // success, old state is not changed
            elem = potentialElem;
            //printf("Pop value %d\n", elem);
            return true;
        }
        return false;
    }
    
    __device__ void push(const T& val) {
        Int16 reservedIndex = -1;
        WAIT(reserveElem(reservedIndex));
        data[reservedIndex] = val;
        memfence(); // make elem valid only after assigning it
        WAIT(makeElemValid(reservedIndex));
        //printf("Push value %d\n", val);
    }
    
    __device__ T pop() {
        T elem;
        WAIT(tryPop(elem));
        return elem;
    }
        
private:
    using UInt64 = unsigned long long;
    
    union State {
        __host__ __device__ State() : full(0) {}
        struct {
            Int16 head; // index of head
            Int16 valid; // number of valid elements
            Int16 reserved; // number of reserved elements (reserved <= valid <= reserved + 1)
        } part;
        UInt64 full;
    } state;

    static_assert(sizeof(Int16) == 2);
    static_assert(sizeof(UInt64) == 8);
    static_assert(sizeof(State) == 8);
    
    ManagedVector<T> data;
};

/*
 * Code of this function is taken from cooperative_groups::details::sync_grids()
 * defined in cooperative_groups/details/sync.h
 */
__forceinline__ __device__
void myCudaBarrier(unsigned int expectedBlocks, volatile unsigned int* arrived, bool master) {
    bool cta_master = (threadIdx.x + threadIdx.y + threadIdx.z == 0);

    __syncthreads();

    if (cta_master) {
        unsigned int nb = 1;
        if (master) {
            nb = 0x80000000 - (expectedBlocks - 1);
        }

        __threadfence();

        unsigned int oldArrive;
        oldArrive = cooperative_groups::details::atomic_add(arrived, nb);

        WAIT(cooperative_groups::details::bar_has_flipped(oldArrive, *arrived));

        //flush barrier upon leaving
        cooperative_groups::details::bar_flush((unsigned int*)arrived);
    }

    __syncthreads();
}
