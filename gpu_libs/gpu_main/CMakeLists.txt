find_package(MPI REQUIRED CXX)

add_library(gpu_main
    main.cu
    )
target_include_directories(gpu_main PUBLIC ${MPI_CXX_INCLUDE_DIRS})
target_link_libraries(gpu_main PUBLIC ${MPI_CXX_LIBRARIES})
target_link_libraries(gpu_main PUBLIC gpumpi_common gpu_libc cuda_btl)
