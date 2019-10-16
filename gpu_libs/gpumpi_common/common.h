#ifndef COMMON_H
#define COMMON_H

#include <iostream>

#define $ std::cerr << "STILL ALIVE: " << __LINE__ << std::endl;


#define CUDA_CHECK(expr) do {\
    cudaError_t err = (expr);\
    if (err != cudaSuccess) {\
        std::cerr << "CUDA ERROR: " << __FILE__ << ":" << __LINE__ << ": " << #expr << " <" << cudaGetErrorName(err) << "> " << cudaGetErrorString(err) << "\n"; \
        abort(); \
    }\
} while(0)

#define MPI_CHECK(expr) \
    if ((expr) != MPI_SUCCESS) { \
        std::cerr << "MPI ERROR: " << __FILE__ << ":" << __LINE__ << ": " << #expr << "\n"; \
        abort(); \
    }

#endif // COMMON_H
