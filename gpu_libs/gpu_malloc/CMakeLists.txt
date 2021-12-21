add_library(gpu_malloc
    dyn_malloc.cu
    warp_malloc.cu
    dynamic_allocator.cu
    combined_malloc.cu
    no_header_malloc.cu
    )
target_link_libraries(gpu_malloc PUBLIC common)
