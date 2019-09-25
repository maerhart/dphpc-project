cmake_minimum_required(VERSION 3.12)

project(examples LANGUAGES C CXX CUDA)

set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

add_executable(cpi cpi.c.cu)
target_link_libraries(cpi gpu_libs)
