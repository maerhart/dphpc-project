add_library(gpu_malloc
    dyn_malloc.cu
    warp_malloc.cu
    dynamic_allocator.cu
    )
target_link_libraries(gpu_malloc PUBLIC common)
