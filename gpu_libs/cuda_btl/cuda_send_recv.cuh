#ifndef CUDA_SEND_RECV
#define CUDA_SEND_RECV

#include <iostream>
#include <cstdlib>
#include <cassert>

#include <cuda.h>
#include <cooperative_groups.h>

#define LOG(fmt, ...) printf("Thread %d " __FILE__ ":%d " fmt "\n", cg::this_grid().thread_rank(), __LINE__,## __VA_ARGS__)

#define ALIVE LOG("STILL ALIVE!");

#define CUDA_CHECK(expr) do {\
    cudaError_t err = (expr);\
    if (err != cudaSuccess) {\
        std::cerr << "CUDA ERROR: " << __FILE__ << ":" << __LINE__ << ": " << #expr << " <" << cudaGetErrorName(err) << "> " << cudaGetErrorString(err) << "\n"; \
        abort(); \
    }\
} while(0)

__device__ void memcpy_volatile(volatile void *dst, volatile void *src, size_t n)
{
    volatile char *d = (volatile char*) dst;
    volatile char *s = (volatile char*) src;
    for (size_t i = 0; i < n; i++) {
        d[i] = s[i];
    }
}

namespace cg = cooperative_groups;

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
        CUDA_CHECK(cudaFree((T*)mData));
    }

    // Can't support [] as in usual std::vector, because
    // all memory accesses to managed memory should be volatile.

    __host__ __device__ void set(int index, const T& value) volatile {
        assert(0 <= index && index < mSize);
        ((volatile T*)mData)[index] = value;
    }

    __host__ __device__ volatile T* get(int index) volatile {
        assert(0 <= index && index < mSize);
        return (volatile T*)(mData + index);
    }

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

template <typename T>
class CircularQueue {
public:
    
    CircularQueue(int size)
        : messages(size)
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

    __host__ __device__ void push(const T& md) volatile {
        int position = bufferState.push();
        messages.set(position, md);
        active.set(position, true);
    }
    
    __host__ __device__ void pop(volatile T* elem) volatile {
        int index = elem - messages.get(0);
        active.set(index, false);
        while (!*active.get(index) && !bufferState.empty()) {
            index = bufferState.pop();
        }
    }

    __host__ __device__ volatile T* head() volatile {
        if (bufferState.empty()) return nullptr;
        return messages.get(bufferState.head);
    }

    __host__ __device__ volatile T* next(volatile T* elem) volatile {
        int index = elem - messages.get(0);
        int nextIndex;
        while (true) {
            nextIndex = (index + 1) % messages.size();
            if (nextIndex == bufferState.tail) return nullptr;
            if (*active.get(nextIndex)) return messages.get(nextIndex);
        }
    }

private:
    ManagedVector<T> messages;
    ManagedVector<bool> active;
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
            volatile MemoryFragment* fragment = fragments.get(i);
            if (fragment->memoryLock.tryLock()) return fragment;
        }
        return nullptr;
    }

     // Try to find free fragment and lock it.
     // If there are no free fragments this thread will wait for it.
//     __host__ __device__ MemoryFragment* lockFreeFragment() {
//         volatile MemoryFragment* fragment = nullptr;
//         while (!fragment) {
//             fragment = tryLockFreeFragment();
//         }
//         return fragment;
//     }

    ManagedVector<MemoryFragment> fragments;
};

} // namespace

#endif
