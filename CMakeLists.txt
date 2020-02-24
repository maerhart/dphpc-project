cmake_minimum_required(VERSION 3.12)

project(gpumpi)

set(CMAKE_CUDA_FLAGS
    "${CMAKE_CUDA_FLAGS} \
    -gencode arch=compute_60,code=sm_60 \
    -gencode arch=compute_61,code=sm_61 \
    -gencode arch=compute_70,code=sm_70")

add_subdirectory(source_converter)
add_subdirectory(gpu_libs)
add_subdirectory(scripts)

enable_testing()
add_subdirectory(tests)
