#include "stdio.cuh"

#include "assert.cuh"

#include "stdarg.cuh"

#include "mp4_printf.cuh"

__device__ __gpu_FILE *__gpu_stdout = (__gpu_FILE*) 1;

__device__ __gpu_FILE *__gpu_stderr = (__gpu_FILE*) 2;

__device__ int __gpu_fprintf(__gpu_FILE * stream, const char * format, ...) {
    va_list arglist;
    va_start(arglist, format);
    int res = __gpu_vfprintf(stream, format, arglist);
    va_end(arglist);
    return res;
}

__device__ int __gpu_fscanf(__gpu_FILE *stream, const char *format, ...) {
    NOT_IMPLEMENTED;
    return 0;
}


__device__ int __gpu_fflush(__gpu_FILE */*__stream*/) {
    // nothing to do
    return 0;
}

__device__ __gpu_FILE *__gpu_fopen(const char *pathname, const char *mode) {
    // TODO not implemented
    // for now pretend that it is not possible to open any file
    return nullptr;
}

__device__ int __gpu_fclose(__gpu_FILE *stream) {
    NOT_IMPLEMENTED;
    return 0;
}

__device__ static char stub[] = "stub";

__device__ char *__gpu_fgets(char *s, int size, __gpu_FILE *stream) {
    NOT_IMPLEMENTED;
    return stub;
}

__device__ int __gpu_putchar(int c) {
    NOT_IMPLEMENTED;
    return 0;
}

__device__ int __gpu_sprintf(char *str, const char *format, ...) {
    va_list arglist;
    va_start(arglist, format);
    const int ret = vsnprintf_(str, (size_t)(-1), format, arglist);
    va_end(arglist);
    return 0;
}

__device__ int __gpu_puts(const char *s) {
    NOT_IMPLEMENTED;
    return 0;
}

__device__ int __gpu_fgetc(__gpu_FILE *stream) {
    NOT_IMPLEMENTED;
    return 0;
}

__device__ int __gpu_feof(__gpu_FILE *stream) {
    NOT_IMPLEMENTED;
    return 0;
}
