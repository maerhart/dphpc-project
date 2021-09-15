#pragma once

#include <cassert>
#include <cstdio>

#include "stdlib.cuh"

namespace CudaMPI {

void initError();
void freeError();
__device__ void recordError(const char* msg);
void printLastError();

}; // namespace

#define STR1(x) #x
#define STR(x) STR1(x)
#define __STR_LINE__ STR(__LINE__)

#ifndef NDEBUG
    #define __gpu_assert(expr) do { \
        if (!(expr)) { \
            CudaMPI::recordError("__gpu_assert FAILED " __FILE__ ":" __STR_LINE__ " '" #expr "'"); \
            __gpu_abort(); \
        } \
    } while (0)
#else
    #define __gpu_assert(expr) // no op
#endif

#define NOT_IMPLEMENTED do { \
    CudaMPI::recordError("NOT_IMPLEMENTED " __FILE__ ":" __STR_LINE__); \
    __gpu_abort(); \
} while (0) 
