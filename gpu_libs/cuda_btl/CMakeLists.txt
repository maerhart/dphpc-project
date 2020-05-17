cmake_minimum_required(VERSION 3.12)

project(cuda_btl LANGUAGES C CXX CUDA)

set(CMAKE_INCLUDE_CURRENT_DIR ON) # all subdirs will be automatically added to include path of targets
set(CMAKE_INCLUDE_CURRENT_DIR_IN_INTERFACE ON) # all subdirs will be automatically added to include path of dependencies

set(CMAKE_CUDA_SEPARABLE_COMPILATION ON) # prevents mess with host/device linking of the functions with the same name

# enable cooperative groups (compute capability >= 6.0)
# enable system-wide atomics: atomicCAS_system (compute capability >= 6.0)
# It is important to incude multiple compute capabilities, because
# sometimes managed memory doesn't on newer devices when compiled with old
# compute capability.
set(CMAKE_CUDA_FLAGS
    "${CMAKE_CUDA_FLAGS} \
    -gencode arch=compute_60,code=sm_60 \
    -gencode arch=compute_61,code=sm_61 \
    -gencode arch=compute_70,code=sm_70")

add_library(cuda_btl cuda_mpi.cu)
#target_compile_definitions(cuda_btl PRIVATE ENABLE_GPU_MPI_LOG)
target_link_libraries(cuda_btl PUBLIC common)
