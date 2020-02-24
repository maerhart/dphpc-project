cmake_minimum_required(VERSION 3.12)

project(gpu_mpi_tests LANGUAGES C CXX CUDA)

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


add_executable(cuda_btl_test cuda_mpi_test.cu)
target_link_libraries(cuda_btl_test cuda_btl)
#target_compile_definitions(cuda_btl_test PRIVATE ENABLE_GPU_MPI_LOG)
target_compile_options(cuda_btl_test PRIVATE -g)
add_test(cuda_btl_test cuda_btl_test)
