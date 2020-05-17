cmake_minimum_required(VERSION 3.12)

project(gpumpi)

set(CMAKE_INCLUDE_CURRENT_DIR ON) # all subdirs will be automatically added to include path of targets
set(CMAKE_INCLUDE_CURRENT_DIR_IN_INTERFACE ON) # all subdirs will be automatically added to include path of dependencies

set(CMAKE_CUDA_FLAGS
    "${CMAKE_CUDA_FLAGS} \
    -gencode arch=compute_60,code=sm_60 \
    -gencode arch=compute_61,code=sm_61 \
    -gencode arch=compute_70,code=sm_70")

add_subdirectory(source_converter)
add_subdirectory(gpu_libs)
add_subdirectory(scripts)
add_subdirectory(common)

enable_testing()
add_subdirectory(tests)
