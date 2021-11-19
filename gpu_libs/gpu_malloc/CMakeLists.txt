add_library(gpu_malloc
    dyn_malloc.cu
    )
target_link_libraries(gpu_malloc PUBLIC common)
