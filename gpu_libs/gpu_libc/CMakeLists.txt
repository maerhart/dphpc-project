add_library(gpu_libc
    stdarg.cu
    gpu_stdio.cu
    gpu_stdlib.cu
    gpu_string.cu
    gpu_ctypes.cu
    gpu_unistd.cu
    mp4_printf.cu
    sys/gpu_time.cu
    errno.cu
    )
target_include_directories(gpu_libc PUBLIC sys)
