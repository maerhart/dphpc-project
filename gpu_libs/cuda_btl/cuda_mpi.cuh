#ifndef CUDA_SEND_RECV
#define CUDA_SEND_RECV

#include <iostream>
#include <cstdlib>
#include <cassert>
#include <type_traits>

#include <cuda.h>
#include <cooperative_groups.h>


namespace cg = cooperative_groups;

#define CUDA_CHECK(expr) do {\
    cudaError_t err = (expr);\
    if (err != cudaSuccess) {\
        std::cerr << "CUDA ERROR: " << __FILE__ << ":" << __LINE__ << ": " << #expr << " <" << cudaGetErrorName(err) << "> " << cudaGetErrorString(err) << "\n"; \
        abort(); \
    }\
} while(0)

#define VOLATILE(x) (*((volatile std::remove_reference_t<decltype(x)>*)&(x)))

__device__ void memcpy_volatile(volatile void *dst, volatile void *src, size_t n);

namespace CudaMPI {

template <typename T>
class DeviceVector {
public:
    __device__ DeviceVector()
        : mSize(0)
        , mReserved(1)
        , mData((T*) malloc(mReserved * sizeof(T)))
    {
    }
    
    __device__ DeviceVector(int n, const T& value = T())
        : mSize(n)
        , mReserved(n)
        , mData((T*) malloc(mReserved * sizeof(T)))
    {
        for (int i = 0; i < mSize; i++) {
            new (&mData[i]) T(value);
        }
    }
    
    __device__ ~DeviceVector() {
        for (int i = 0; i < mSize; i++) {
            mData[i].~T();
        }
        free(mData);
    }
    
    __device__ T& operator[] (int index) { return mData[index]; }
    
    // TODO: remove this
    __device__ T& operator[] (int index) volatile { return (T&)mData[index]; }
    
    __device__ void resize(int new_size) {
        if (new_size == mSize) return;
        if (new_size > mSize) {
            if (new_size > mReserved) {
                reserve(new_size);
            }
            for (int i = mSize; i < new_size; i++) {
                new (&mData[i]) T();
            }
        }
        if (new_size < mSize) {
            for (int i = new_size; i < mSize; i++) {
                mData[i].~T();
            }
        }
        mSize = new_size;
    }
    
    __device__ void reserve(int new_reserve) {
        if (new_reserve == 0) new_reserve = 1; 
        if (new_reserve < mSize) return;
        if (new_reserve == mReserved) return;
        
        T* new_data = (T*)malloc(mReserved * sizeof(T));
        memcpy(new_data, mData, mSize * sizeof(T));
        free(mData);
        mData = new_data;
        mReserved = new_reserve;
    }
    
    __device__ void push_back(const T& val) {
        if (mSize == mReserved) {
            reserve(mReserved * 2);
        }
        new (&mData[mSize++]) T(val);
    }
    
    // TODO: remove volatile
    __device__ int size() const volatile { return mSize; }
    
private:
    __device__ void operator=(const DeviceVector&) = delete;
    __device__ DeviceVector(const DeviceVector&) = delete;
    
    int mSize;
    int mReserved;
    T* mData;
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

    // Can't support [] as in usual std::vector, because
    // all memory accesses to managed memory should be volatile.
    
    __host__ __device__ volatile T& operator[](int index) volatile {
        assert(0 <= index && index < mSize);
        return VOLATILE(mData[index]);
    }

//     __host__ __device__ void set(int index, const T& value) volatile {
//         assert(0 <= index && index < mSize);
//         ((volatile T*)mData)[index] = value;
//     }
// 
//     __host__ __device__ volatile T* get(int index) volatile {
//         assert(0 <= index && index < mSize);
//         return (volatile T*)(mData + index);
//     }

    __host__ __device__ int size() const volatile { return mSize; }

private:
    int mSize;
    T* mData;
};


namespace THREAD_LOCALITY {
    enum Type {
        WARP,
        BLOCK,
        GRID,
        MULTI_GRID
    };
}

namespace MEMORY_LOCALITY {
    enum Type {
        GLOBAL,
        LOCAL,
        SHARED,
        CONST,
        OTHER
    };
}

namespace cg = cooperative_groups;

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

inline __host__ __device__ void cudaGlobalFence() {
    #if defined(__CUDA_ARCH__)
        __threadfence_system();
    #else
        __sync_synchronize();
    #endif
}

class ManagedMemoryLock {
public:
    ManagedMemoryLock() : locked(0) {
    }

    __host__ __device__ bool tryLock() volatile {
        // Since CAS returns old value, the operation is successful
        // if an old value (second arg of CAS) equal to the return value
        bool success = false;
        #if defined(__CUDA_ARCH__)
            success = (0 == atomicCAS_system((unsigned*)&locked, 0, 1));
        #else
            success = (0 == __sync_val_compare_and_swap((unsigned*)&locked, 0, 1));
        #endif
        return success;
    }

    __host__ __device__ void lock() volatile {
        while (!tryLock()) {}
    }

    __host__ __device__ void unlock() volatile {
        cudaGlobalFence();
        locked = 0;
    }
private:
    unsigned locked;
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

struct MemoryFragment {
    MemoryFragment(int size)
        : data(size)
        , ownerProcess(-1) // will be defined by the process that locks this fragments
    {}

    // this lock is shared between pair of processes
    // that use this memory fragment for communication.
    // It means that when this MemoryFragment is locked,
    // sender and receiver doesn't unlock it, but
    // use "ownerProcess" field to understand
    // what process should use memory at each time moment
    ManagedMemoryLock memoryLock;
    volatile int ownerProcess;

    ManagedVector<char> data;
};

struct SharedFragmentBuffer {
    SharedFragmentBuffer(int numFragments, int fragmentSize)
        : fragments(numFragments, fragmentSize)
    {
    }

    // Try to find free fragment and lock it.
    // Return nullptr if there are no free fragments.
    __host__ __device__ volatile MemoryFragment* tryLockFreeFragment() {
        for (int i = 0; i < fragments.size(); i++) {
            volatile MemoryFragment& fragment = fragments[i];
            if (fragment.memoryLock.tryLock()) {
                return &fragment;
            }
        }
        return nullptr;
    }

    ManagedVector<MemoryFragment> fragments;
};

enum { ANY_SRC = -1 };
enum { ANY_TAG = -1 };

struct PendingOperation {
    enum class Type { SEND, RECV };

    // one of two state transitions are possible
    // STARTED -> POSTED -> SYNCED -> COMPLETED
    // STARTED -> MATCHED -> ALLOCATED -> SYNCED -> COMPLETED
    enum class State {
        STARTED,
        POSTED,
        MATCHED,
        ALLOCATED,
        SYNCED,
        COMPLETED
    };

    Type type = Type::SEND;
    State state = State::STARTED;
    volatile MemoryFragment* fragment = nullptr;
    int otherThread = 0;
    PendingOperation* foreignPendingOperation = nullptr;
    void* data = nullptr;
    int count = 0;
    int comm = 0;
    int tag = 0;
    bool canBeFreed = false;
    //bool unused = true;
    
    //__device__ void free() { unused = true; }
};

// This class is used for tracking skipped operations in STARTED state.
// It helps to prevent breaking the standardized order of MPI operations.
class ProgressState {
public:
    __device__ ProgressState()
        : mStartedSendSkip(false)
        , mStartedRecvSkip(false)
    {
    }
    
    // TODO: Current implementation is too simplistic and conservative,
    // it can be improved later by considering numbers of src and dst threads.
    __device__ void markStartedSendSkip(int /*dst*/) { mStartedSendSkip = true; }
    __device__ bool isStartedSendSkip(int /*dst*/) { return mStartedSendSkip; }
    
    __device__ void markStartedRecvSkip(int /*src*/) { mStartedRecvSkip = true; }
    __device__ bool isStartedRecvSkip(int /*src*/) { return mStartedRecvSkip; }
private:
    bool mStartedSendSkip;
    bool mStartedRecvSkip;
};

__device__ void progress();

__device__ void progressSend(PendingOperation& send, ProgressState& state);
__device__ void progressRecv(PendingOperation& recv, ProgressState& state);

__device__ void progressStartedRecv(PendingOperation& recv, ProgressState& state);
__device__ void progressPostedRecv(PendingOperation& recv);
__device__ void progressMatchedRecv(PendingOperation& recv);
__device__ void progressAllocatedRecv(PendingOperation& recv);
__device__ void progressSyncedRecv(PendingOperation& recv);
__device__ void progressCompletedRecv(PendingOperation& recv);

__device__ void progressStartedSend(PendingOperation& send, ProgressState& state);
__device__ void progressPostedSend(PendingOperation& send);
__device__ void progressMatchedSend(PendingOperation& send);
__device__ void progressAllocatedSend(PendingOperation& send);
__device__ void progressSyncedSend(PendingOperation& send);
__device__ void progressCompletedSend(PendingOperation& send);

struct ThreadPrivateState {

    struct Context {
        int pendingBufferSize;
    };

    __device__ PendingOperation* allocatePendingOperation();

    using PendingOperations = CircularQueue<PendingOperation, DeviceVector>;
    __device__ PendingOperations& getPendingOperations() { return pendingOperations; }

    struct Holder {
        __device__ Holder(const Context& ctx);
        __device__ ~Holder();
    };

private:

    __device__ explicit ThreadPrivateState(const Context& ctx)
        : pendingOperations(ctx.pendingBufferSize)
    {
    }

    PendingOperations pendingOperations;
};

struct MessageDescriptor {
    PendingOperation* privatePointer;
    int src;
    int comm;
    int tag;

    __host__ __device__ volatile MessageDescriptor& operator=(const MessageDescriptor& other) volatile {
        privatePointer = other.privatePointer;
        src = other.src;
        comm = other.comm;
        tag = other.tag;
        return *this;
    }
};

struct IncomingFragment {
    volatile MemoryFragment* volatile fragment;
    PendingOperation* privatePointer;

    __host__ __device__ volatile IncomingFragment& operator=(const IncomingFragment& other) volatile {
        fragment = other.fragment;
        privatePointer = other.privatePointer;
        return *this;
    }
};

struct SharedThreadState {
    SharedThreadState(int recvListSize, int numIncomingFragments)
        : unexpectedRecv(recvListSize)
        , expectedRecv(recvListSize)
        , incomingFragments(numIncomingFragments)
    {}

    ManagedMemoryLock recvLock;
    CircularQueue<MessageDescriptor, ManagedVector> unexpectedRecv;
    CircularQueue<MessageDescriptor, ManagedVector> expectedRecv;

    ManagedMemoryLock fragLock;
    CircularQueue<IncomingFragment, ManagedVector> incomingFragments;
};

class SharedState {
public:
    struct Context {
        int numThreads;
        int recvListSize;
        int numFragments;
        int fragmentSize;
        int numIncomingFragments;
    };

private:
    SharedState(const Context& ctx)
        : sharedThreadState(ctx.numThreads, ctx.recvListSize, ctx.numIncomingFragments)
        , sharedFragmentBuffer(ctx.numFragments, ctx.fragmentSize)
    {
    }

public:
    struct Holder {
        Holder(const Context& ctx) {
            CUDA_CHECK(cudaMallocManaged(&sharedState, sizeof(SharedState)));
            new (sharedState) SharedState(ctx);
            assert(sharedState);
        }
        ~Holder() {
            sharedState->~SharedState();
            CUDA_CHECK(cudaFree(sharedState));
        }
        SharedState* get() const { return sharedState; }
    private:
        SharedState* sharedState;
    };

    ManagedVector<SharedThreadState> sharedThreadState;
    SharedFragmentBuffer sharedFragmentBuffer;
};

__device__ SharedState& sharedState();

__device__ void setSharedState(SharedState* sharedState);

__device__ ThreadPrivateState& threadPrivateState();

__device__ PendingOperation* isend(int dst, const void* data, int count, int comm, int tag);

__device__ PendingOperation* irecv(int src, void* data, int count, int comm, int tag);

__device__ void receiveFragmentPointers();

__device__ void progress();

__device__ bool test(PendingOperation* op);

__device__ void wait(PendingOperation* op);

__device__ void initialize();

} // namespace


#endif
