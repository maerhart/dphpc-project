cmake_minimum_required(VERSION 3.12)

project(gpu_libs LANGUAGES C CXX CUDA)

set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# enable cooperative groups (compute capability >= 6.0)
# enable system-wide atomics: atomicCAS_system (compute capability >= 6.0)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_60")

add_library(managed_memory
    managed_memory.cu
    managed_memory.h
    )

add_executable(test_managed_memory test_managed_memory.cu)
target_link_libraries(test_managed_memory PUBLIC managed_memory)
