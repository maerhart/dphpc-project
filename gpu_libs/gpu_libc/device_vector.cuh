#ifndef DEVICE_VECTOR_CUH
#define DEVICE_VECTOR_CUH

#include "assert.cuh"
#include "common.h"

namespace CudaMPI {

template <typename T>
class DeviceVector {
private:
    static __device__ T* checked_alloc(size_t elems) {
        size_t nbytes = elems * sizeof(T);
        T* ptr = (T*) __gpu_malloc(nbytes);
        __gpu_assert(ptr);
        return ptr;
    }
public:
    __device__ DeviceVector()
        : mSize(0)
        , mReserved(1)
        , mData(DeviceVector<T>::checked_alloc(mReserved))
    {
    }
    
    __device__ DeviceVector(int n, const T& value = T())
        : mSize(n)
        , mReserved(n > 0 ? n : 1)
        , mData(DeviceVector<T>::checked_alloc(mReserved))
    {
        for (int i = 0; i < mSize; i++) {
            new (&mData[i]) T(value);
        }
    }
    
    __device__ DeviceVector(const DeviceVector& other)
        : mSize(0)
        , mReserved(other.size())
        , mData(DeviceVector<T>::checked_alloc(mReserved))
    {
        for (int i = 0; i < other.size(); i++) {
            mData[i] = other[i];
        }
    }
    
    // Yes, I know that this is inefficient.
    // Have time to fix? Do it.
    __device__ void operator=(DeviceVector other) {
        resize(other.size());
        for (int i = 0; i < other.size(); i++) {
            mData[i] = other[i];
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
        
        T* new_data = DeviceVector<T>::checked_alloc(new_reserve);
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
    
    __host__ __device__ T& operator[](int index) {
        assert(0 <= index && index < mSize);
        return mData[index];
    }

    __host__ __device__ int size() const { return mSize; }

private:
    int mSize;
    T* mData;
};

} // namespace

#endif
