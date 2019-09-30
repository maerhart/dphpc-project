cmake_minimum_required(VERSION 3.12)

project(gpu_libs LANGUAGES C CXX CUDA)

set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

find_package(MPI REQUIRED)

add_library(gpu_libc
    gpu_libc/gpu_stdio.cu
    gpu_libc/gpu_stdlib.cu
    gpu_libc/gpu_string.cu
    gpu_libc/gpu_unistd.cu
    gpu_libc/sys/gpu_time.cu
    )
target_include_directories(gpu_libc PUBLIC gpu_libc)

add_library(gpu_mpi gpu_mpi/mpi.cu)
target_include_directories(gpu_mpi PUBLIC gpu_mpi)

add_library(gpu_main gpu_main/main.cu)
target_include_directories(gpu_main PRIVATE ${MPI_C_INCLUDE_DIRS})
target_link_libraries(gpu_main ${MPI_C_LIBRARIES})

add_library(gpu_libs INTERFACE)
target_link_libraries(gpu_libs INTERFACE gpu_libc gpu_mpi gpu_main)
