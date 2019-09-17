cmake_minimum_required(VERSION 3.12)

project(examples LANGUAGES C CXX CUDA)

set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

add_executable(cpi cpi.c.cu)
#target_include_directories(cpi PUBLIC ${MPI_C_INCLUDE_DIRS})
#target_link_libraries(cpi ${MPI_C_LIBRARIES})
target_link_libraries(cpi gpu_libs)
