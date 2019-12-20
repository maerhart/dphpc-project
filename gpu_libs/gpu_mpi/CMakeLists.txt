add_library(gpu_mpi
    mpi.cu
    )
target_link_libraries(gpu_mpi PUBLIC gpu_libc gpumpi_common)
target_link_libraries(gpu_mpi PRIVATE cuda_btl)
