cmake_minimum_required(VERSION 3.12)

project(gpu_libs LANGUAGES C CXX CUDA)

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
    -gencode arch=compute_75,code=sm_75")

add_executable(cuda_p2p_benchmark cuda_p2p_benchmark.cu)

add_executable(cuda_p2p_warp cuda_p2p_warp.cu)
add_executable(cuda_p2p_block cuda_p2p_block.cu)
add_executable(cuda_p2p_device cuda_p2p_device.cu)
add_executable(cuda_p2p_host_device cuda_p2p_host_device.cu)
add_executable(cuda_p2p_multi_device cuda_p2p_multi_device.cu)


configure_file(run_benchmarks.sh run_benchmarks.sh COPYONLY)
