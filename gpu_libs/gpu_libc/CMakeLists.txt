add_library(gpu_libc
    stdarg.cu
    stdio.cu
    stdlib.cu
    string.cu
    stdint.cu
    ctype.cu
    unistd.cu
    mp4_printf.cu
    sys/time.cu
    errno.cu
    omp.cu
    )
target_include_directories(gpu_libc PUBLIC sys)
target_link_libraries(gpu_libc PUBLIC cuda_btl)
