#ifndef CUDA_SEND_RECV
#define CUDA_SEND_RECV

#include <iostream>
#include <cstdlib>
#include <cassert>

#include <cuda.h>
#include <cooperative_groups.h>

#define CUDA_CHECK(expr) do {\
    cudaError_t err = (expr);\
    if (err != cudaSuccess) {\
        std::cerr << "CUDA ERROR: " << __FILE__ << ":" << __LINE__ << ": " << #expr << " <" << cudaGetErrorName(err) << "> " << cudaGetErrorString(err) << "\n"; \
        abort(); \
    }\
} while(0)

__device__ void memcpy_volatile(volatile void *dst, volatile void *src, size_t n)
{
    volatile char *d = (char*) dst;
    volatile char *s = (char*) src;
    for (size_t i = 0; i < n; i++) {
        d[i] = s[i];
    }   
}

namespace CudaMPI {
    
template <typename T>
class Vector {
public:
    __device__ Vector()
        : mSize(0)
        , mReserved(1)
        , mData((T*) malloc(mReserved * sizeof(T)))
    {
    }
    
    __device__ Vector(int n)
        : mSize(n)
        , mReserved(n)
        , mData((T*) malloc(mReserved * sizeof(T)))
    {
        for (int i = 0; i < mSize; i++) {
            new (&mData[i]) T();
        }
    }
    
    __device__ ~Vector() {
        for (int i = 0; i < mSize; i++) {
            mData[i].~T();
        }
        free(mData);
    }
    
    __device__ T& operator[] (int index) { return mData[index]; }
    
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
    
    __device__ int size() const { return mSize; }
    
    __device__ void disorderedRemove(int index) {
        assert(0 <= index && index < mSize);
        // swap with last and remove
        if (mSize == 1) {
            mData[index].~T();
            mSize = 0;
        } else {
            mData[index].~T();
            memcpy(&mData[index], &mData[mSize - 1], sizeof(T));
            mSize--;
        }
    }
    
private:
    __device__ Vector operator=(const Vector&) = delete;
    __device__ Vector(const Vector&) = delete;
    
    int mSize;
    int mReserved;
    T* mData;
};

template <typename T>
class ManagedVector {
public:
    template <typename... Args>
    ManagedVector(int size, Args... args) : mSize(size)
    {
        CUDA_CHECK(cudaMallocManaged(&mData, mSize * sizeof(T)));
        for (int i = 0; i < size; i++) {
            new (&mData[i]) T(args...);
        }
    }

    ~ManagedVector() {
        for (int i = 0; i < mSize; i++) {
            mData[i].~T();
        }
        CUDA_CHECK(cudaFree(mData));
    }
    
    __host__ __device__ T& operator[] (int index) {
        return mData[index];
    }
    
    __host__ __device__ int size() const { return mSize; }

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

// check where memory is located
// from https://stackoverflow.com/questions/42519766/can-i-check-whether-an-address-is-in-shared-memory
#define DEFINE_is_xxx_memory(LOCATION) \
    __device__ bool is_ ## LOCATION ## _memory(void *ptr) {\
        int res;\
        asm("{"\
            ".reg .pred p;\n\t"\
            "isspacep." #LOCATION " p, %1;\n\t"\
            "selp.b32 %0, 1, 0, p;\n\t"\
            "}"\
            : "=r"(res): "l"(ptr));\
        return res;\
    }

DEFINE_is_xxx_memory(global) // __device__, __managed__, malloc() from kernel
DEFINE_is_xxx_memory(local) // scope-local stack variables
DEFINE_is_xxx_memory(shared) // __shared__
DEFINE_is_xxx_memory(const) // __constant__





class CudaSendRecv {
public:

    struct Location {
        int gridIndex;
        int blockIndex;
        int warpIndex;
    };
    __device__ int gridRank(int threadRankInMultiGrid) {
        return threadRankInMultiGrid / cg::this_grid().size();
    }
    __device__ int rankInGrid(int threadRankInMultiGrid) {
        return threadRankInMultiGrid % cg::this_grid().size();
    }
    __device__ int blockRank(int threadRankInMultiGrid) {
        int threadRankInGrid = rankInGrid(threadRankInMultiGrid);
        return threadRankInGrid / cg::this_thread_block().size();
    }
    __device__ int rankInBlock(int threadRankInMultiGrid) {
        int threadRankInGrid = rankInGrid(threadRankInMultiGrid);
        return threadRankInGrid % cg::this_thread_block().size();
    }
    __device__ int warpRank(int threadRankInMultiGrid) {
        int threadRankInBlock = rankInBlock(threadRankInMultiGrid);
        return threadRankInBlock / warpSize;
    }
    __device__ int rankInWarp(int threadRankInMultiGrid) {
        int threadRankInBlock = rankInBlock(threadRankInMultiGrid);
        return threadRankInBlock % warpSize;
    }

    __device__ THREAD_LOCALITY::Type getThreadLocalityType(int threadA, int threadB) {
        if (gridRank(threadA) != gridRank(threadB)) {
            return THREAD_LOCALITY::MULTI_GRID;
        } else if (blockRank(threadA) != blockRank(threadB)) {
            return THREAD_LOCALITY::GRID;
        } else if (warpRank(threadA) != warpRank(threadB)) {
            return THREAD_LOCALITY::BLOCK;
        } else {
            return THREAD_LOCALITY::WARP;
        }
    }

    __device__ MEMORY_LOCALITY::Type getMemoryLocalityType(void* ptr) {
        if (is_global_memory(ptr)) {
            return MEMORY_LOCALITY::GLOBAL;
        } else if (is_local_memory(ptr)) {
            return MEMORY_LOCALITY::LOCAL;
        } else if (is_shared_memory(ptr)) {
            return MEMORY_LOCALITY::SHARED;
        } else if (is_const_memory(ptr)) {
            return MEMORY_LOCALITY::CONST;
        } else {
            return MEMORY_LOCALITY::OTHER;
        }
    }

    __device__ void sendRecv(void* ptr, int n, int srcThread, int dstThread) {
        int thisThread = cg::this_multi_grid().thread_rank();
        if (thisThread != srcThread && thisThread != dstThread) return;

        THREAD_LOCALITY::Type threadLocality = getThreadLocalityType(srcThread, dstThread);
        MEMORY_LOCALITY::Type memoryLocality = getMemoryLocalityType(ptr);

        // check one of two: memory

        switch (threadLocality) {
            case THREAD_LOCALITY::WARP:
                sendRecvWarp(ptr, n, srcThread, dstThread);
                break;
            case THREAD_LOCALITY::BLOCK:
                sendRecvBlock(ptr, n, srcThread, dstThread);
                break;
            case THREAD_LOCALITY::GRID:
                sendRecvGrid(ptr, n, srcThread, dstThread);
                break;
            case THREAD_LOCALITY::MULTI_GRID:
                sendRecvMultiGrid(ptr, n, srcThread, dstThread);
                break;
        }
    }

    __device__ void sendRecvWarp(void* ptr, int n, int srcThread, int dstThread);
    __device__ void sendRecvBlock(void* ptr, int n, int srcThread, int dstThread);
    __device__ void sendRecvGrid(void* ptr, int n, int srcThread, int dstThread);
    __device__ void sendRecvMultiGrid(void* ptr, int n, int srcThread, int dstThread);

};



template <typename T>
struct LocalVector {
    template <typename... Args>
    LocalVector(int size, Args... args) : size(size)
    {
        CUDA_CHECK(cudaMalloc(&data, size * sizeof(T)));
        for (int i = 0; i < size; i++) {
            new (&data[i]) T(args...);
        }
    }

    ~LocalVector() {
        for (int i = 0; i < size; i++) {
            data[i].~T();
        }
        CUDA_CHECK(cudaFree(data));
    }

    __host__ __device__ T& operator[] (int index) {
        return data[index];
    }

    int size;
    T* data;
};

struct CircularBufferState {
    __host__ __device__ CircularBufferState(int size)
        : size(size)
        , used(0)
        , head(0)
        , tail(0)
    {
    }

    __host__ __device__ bool empty() const { return used == 0; }
    __host__ __device__ bool full() const { return used == size; }

    // reserve and return position for new element at the tail of queue
    __host__ __device__ int push() {
        assert(!full());
        used += 1;
        int position = tail;
        tail = (position + 1) % size;
        return position;
    }

    // release and return (released) position of element from the head of queue
    __host__ __device__ int pop() {
        assert(!empty());
        used -= 1;
        int position = head;
        head = (position + size - 1) % size;
        return position;
    }

    int used;
    int head; // first
    int tail; // next after last
    int size;
};

__host__ __device__ void cudaGlobalFence() {
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

    __host__ __device__ bool tryLock() {
        // Since CAS returns old value, the operation is successful
        // if an old value (second arg of CAS) equal to the return value
        bool success = false;
        #if defined(__CUDA_ARCH__)
            success = (0 == atomicCAS_system(&locked, 0, 1));
        #else
            success = (0 == __sync_val_compare_and_swap(&locked, 0, 1));
        #endif
        return success;
    }

    __host__ __device__ void lock() {
        while (!tryLock()) {}
    }

    __host__ __device__ void unlock() {
        cudaGlobalFence();
        *((volatile unsigned*)&locked) = 0;
    }
private:
    unsigned locked;
};

template <typename T>
class CircularQueue {
public:
    
    CircularQueue(int size)
        : messages(size)
        , active(size, false)
        , bufferState(size)
    {
    }
    
    __host__ __device__ int size() {
        return bufferState.size;
    }
    
    __host__ __device__ bool empty() {
        return bufferState.empty();
    }
    
    __host__ __device__ int full() {
        return bufferState.full();
    }

    __host__ __device__ void push(const T& md) {
        int position = bufferState.push();
        messages[position] = md;
        active[position] = true;
    }
    
    __host__ __device__ void pop(T* elem) {
        int index = elem - &messages[0];
        active[index] = false;
        while (!active[index] && !bufferState.empty()) {
            index = bufferState.pop();
        }
    }

    __host__ __device__ T* head() {
        if (bufferState.empty()) return nullptr;
        return &messages[bufferState.head];
    }

    __host__ __device__ T* next(T* elem) {
        int index = elem - &messages[0];
        int nextIndex;
        while (true) {
            nextIndex = (index + 1) % messages.size();
            if (nextIndex == bufferState.tail) return nullptr;
            if (active[nextIndex]) return &messages[nextIndex];
        }
    }

private:
    ManagedVector<T> messages;
    ManagedVector<bool> active;
    CircularBufferState bufferState;
};

struct ProcessMessageQueues {
public:
    ProcessMessageQueues(int messageQueueSize)
//         : unexpected(messageQueueSize)
//         , receive(messageQueueSize)
    {

    }

    __host__ __device__ void lock() {
        memoryLock.lock();
    }

    __host__ __device__ void unlock() {
        memoryLock.unlock();
    }

//     CircularQueue<MessageEnvelope> unexpected;
//     CircularQueue<MessageEnvelope> receive;

private:
    ManagedMemoryLock memoryLock;
};

struct SharedProcessQueues {
    SharedProcessQueues(int numProcesses, int messageQueueSize)
        : messageQueues(numProcesses, messageQueueSize)
    {
    }
    ManagedVector<ProcessMessageQueues> messageQueues;
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

struct MessageEnvelope {
    enum {
        ANY_SOURCE = -1,
        ANY_TAG = -1
    };

    __host__ __device__ bool match(int source, int destination, int tag, int communicator) {
        if (source != mSource && mSource != ANY_SOURCE && source != ANY_SOURCE) {
            return false;
        }
        if (mDestination != destination) return false;
        if (mTag != tag) return false;
        if (mCommunicator != communicator) return false;
        return true;
    }

    int mSource;
    int mDestination;
    int mTag;
    int mCommunicator;

    bool mMatched;
    MemoryFragment* mMemoryFragment;
    void* mLocalPointer;
    int mSize;
};

struct SharedFragmentBuffer {
    SharedFragmentBuffer(int numFragments, int fragmentSize)
        : fragments(numFragments, fragmentSize)
    {
    }

    // Try to find free fragment and lock it.
    // Return nullptr if there are no free fragments.
    __host__ __device__ MemoryFragment* tryLockFreeFragment() {
        for (int i = 0; i < fragments.size(); i++) {
            MemoryFragment* fragment = &fragments[i];
            if (fragment->memoryLock.tryLock()) return fragment;
        }
        return nullptr;
    }

     // Try to find free fragment and lock it.
     // If there are no free fragments this thread will wait for it.
    __host__ __device__ MemoryFragment* lockFreeFragment() {
        MemoryFragment* fragment = nullptr;
        while (!fragment) {
            fragment = tryLockFreeFragment();
        }
        return fragment;
    }

    ManagedVector<MemoryFragment> fragments;
};


struct MessageRequest {
    MessageRequest() : localBuffer(nullptr) {

    }

    void setLocalBuffer(const void* value) {
        localBuffer = (void*)value;
    }

    enum class Type { SEND, RECV };
    Type type;

    enum class Stage { ONE, TWO, ROUNDTRIP };
    Stage stage;

    MemoryFragment* memoryFragment = nullptr;
    void* localBuffer;
    int bytesLeft;
};

struct LocalFragmentMapping {
    MemoryFragment** localAddress;
    MemoryFragment* memoryFragment;
};

struct MessageSendRecv {

    MessageSendRecv()
        : sharedProcessQueues(1024, 1024)
        , sharedFragmentBuffer(1024, 1024)
    {

    }

    MessageEnvelope* findPostedRecv(int dst, int tag, int comm);
    MessageEnvelope* findUnexpectedSend(int src, int tag, int comm);

    void addToUnexpected(int dst, int tag, int comm);
    void addToRecv(int src, int tag, int comm);

    int currentProcessId();

    void lockMatchingLists(int process);
    void unlockMatchingLists(int process);

    MemoryFragment* allocateMemoryFragment();
    void releaseMemoryFragment(MemoryFragment* mf);

    void send(const void* buf, int count, int dst, int tag, int comm, MessageRequest& request) {
        lockMatchingLists(dst);
        if (MessageEnvelope* postedRecv = findPostedRecv(dst, tag, comm)) {
            // Specify concrete values if they were wildcard
            postedRecv->mSource = currentProcessId();
            postedRecv->mCommunicator = comm;

            postedRecv->mMemoryFragment = allocateMemoryFragment();

            request.memoryFragment = postedRecv->mMemoryFragment;
        } else {
            addToUnexpected(dst, tag, comm);
        }
        unlockMatchingLists(dst);

        request.type = MessageRequest::Type::SEND;
        request.setLocalBuffer(buf);
        request.bytesLeft = count;
    }

    void recv(void* buf, int count, int src, int tag, int comm, MessageRequest& request) {
        lockMatchingLists(currentProcessId());
        if (MessageEnvelope* unexpectedSend = findUnexpectedSend(src, tag, comm)) {
            unexpectedSend->mMemoryFragment = allocateMemoryFragment();

        } else {
            addToRecv(src, tag, comm);
        }
        unlockMatchingLists(currentProcessId());
    }


    SharedProcessQueues sharedProcessQueues;
    SharedFragmentBuffer sharedFragmentBuffer;
};


} // namespace

#endif
