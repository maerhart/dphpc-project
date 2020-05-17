#ifndef COMMON_H
#define COMMON_H

#include <iostream>

#define $ printf("STILL ALIVE: %d\n", __LINE__);


#define CUDA_CHECK(expr) do {\
    cudaError_t err = (expr);\
    if (err != cudaSuccess) {\
        std::cerr << "CUDA ERROR: " << __FILE__ << ":" << __LINE__ << ": " << #expr << " <" << cudaGetErrorName(err) << "> " << cudaGetErrorString(err) << "\n"; \
        abort(); \
    }\
} while(0)


#define MPI_CHECK(expr) \
    if ((expr) != MPI_SUCCESS) { \
        std::cerr << "MPI ERROR (on host): " << __FILE__ << ":" << __LINE__ << ": " << #expr << "\n"; \
        abort(); \
    }
    
#define MPI_CHECK_DEVICE(expr) \
    if ((expr) != MPI_SUCCESS) { \
        printf("MPI ERROR (on device): " __FILE__ ":%d: "  #expr  "\n", __LINE__); \
        asm("trap;"); \
    }

#define VOLATILE(x) (*((volatile std::remove_reference_t<decltype(x)>*)&(x)))

__forceinline__
__host__ __device__ void memcpy_volatile(volatile void *dst, volatile void *src, size_t n)
{
    volatile char *d = (volatile char*) dst;
    volatile char *s = (volatile char*) src;
    for (size_t i = 0; i < n; i++) {
        d[i] = s[i];
    }
}

template <typename T>
class ScopeGuard {
public:
    __host__ __device__ ScopeGuard(T func) : run(true), func(func) {}
    __host__ __device__ ScopeGuard(ScopeGuard<T>&& rhs)
        : run(rhs.run)
        , func(std::move(rhs.func))
    { rhs.run = false; }
    __host__ __device__ ~ScopeGuard() { if (run) func(); }
    __host__ __device__ void commit() { run = false; }
private:
    ScopeGuard(const ScopeGuard<T>& rhs) = delete;
    void operator=(const ScopeGuard<T>& rhs) = delete;
    
    bool run;
    T func;
};

template <typename T>
__host__ __device__ ScopeGuard<T> makeScopeGuard(T func) {
    return ScopeGuard<T>(func);
}
    
#endif // COMMON_H
