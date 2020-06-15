cmake_minimum_required(VERSION 3.12)

project(gpu_libs LANGUAGES C CXX CUDA)

set(CMAKE_INCLUDE_CURRENT_DIR ON) # all subdirs will be automatically added to include path of targets
set(CMAKE_INCLUDE_CURRENT_DIR_IN_INTERFACE ON) # all subdirs will be automatically added to include path of dependencies

set(CMAKE_CUDA_SEPARABLE_COMPILATION ON) # prevents mess with host/device linking of the functions with the same name

# enable cooperative groups (compute capability >= 6.0)
# enable system-wide atomics: atomicCAS_system (compute capability >= 6.0)
set(CMAKE_CUDA_FLAGS 
    "${CMAKE_CUDA_FLAGS} \
    -gencode arch=compute_60,code=sm_60 \
    -gencode arch=compute_61,code=sm_61 \
    -gencode arch=compute_70,code=sm_70")

add_subdirectory(cuda_btl)

add_subdirectory(gpu_libc)
add_subdirectory(gpu_mpi)
add_subdirectory(gpu_main)

add_library(gpu_libs INTERFACE)
target_link_libraries(gpu_libs INTERFACE gpu_libc gpu_mpi gpu_main)

