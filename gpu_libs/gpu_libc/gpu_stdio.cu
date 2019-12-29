#include "stdio.h.cuh"

#include "assert.h.cuh"

#include "stdarg.h.cuh"

__device__ FILE *stdout = (FILE*) 1;

__device__ FILE *stderr = (FILE*) 2;

__device__ int fprintf(FILE * stream, const char * format, ...) {
    va_list arglist;
    va_start(arglist, format);
    int res = vfprintf(stream, format, arglist);
    va_end(arglist);
    return res;
}

__device__ int fflush(FILE */*__stream*/) {
    // nothing to do
    return 0;
}

__device__ FILE *fopen(const char *pathname, const char *mode) {
    NOT_IMPLEMENTED
    return stdout;
}

__device__ int fclose(FILE *stream) {
    NOT_IMPLEMENTED
    return 0;
}

__device__ static char stub[] = "stub";

__device__ char *fgets(char *s, int size, FILE *stream) {
    NOT_IMPLEMENTED
    return stub;
}

__device__ int putchar(int c) {
    NOT_IMPLEMENTED
    return 0;
}

__device__ int sprintf(char *str, const char *format, ...) {
    NOT_IMPLEMENTED
    return 0;
}

__device__ int puts(const char *s) {
    NOT_IMPLEMENTED
    return 0;
}

__device__ int fgetc(FILE *stream) {
    NOT_IMPLEMENTED
    return 0;
}

__device__ int feof(FILE *stream) {
    NOT_IMPLEMENTED
    return 0;
}
