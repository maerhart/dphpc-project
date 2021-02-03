find_package(MPI REQUIRED CXX)

add_library(libc_processor
    libc_processor.cu
    device_host_comm.cu
    global_vars.cu
    )
target_link_libraries(libc_processor PRIVATE common cuda_btl)

add_library(gpu_main main.cu)
target_include_directories(gpu_main PUBLIC ${MPI_CXX_INCLUDE_DIRS})
target_link_libraries(gpu_main PUBLIC ${MPI_CXX_LIBRARIES})
target_link_libraries(gpu_main PRIVATE common gpu_libc libc_processor)
#target_compile_options(gpu_main PRIVATE -g -G)
