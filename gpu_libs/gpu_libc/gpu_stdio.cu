#include "stdio.h.cuh"


__device__ FILE *stdout;

__device__ FILE *stderr;

__device__ int fprintf(FILE * __stream, const char * __format, ...) { return 0; }

__device__ int fflush(FILE *__stream) { return 0; }

__device__ FILE *fopen(const char *pathname, const char *mode) { return stdout; }

__device__ int fclose(FILE *stream) { return 0; }

__device__ static char stub[] = "stub";

__device__ char *fgets(char *s, int size, FILE *stream) { return stub; }

__device__ int putchar(int c) { return 0; }

__device__ int sprintf(char *str, const char *format, ...) { return 0; }

__device__ int puts(const char *s) { return 0; }

__device__ int fgetc(FILE *stream) { return 0; }

__device__ int feof(FILE *stream) { return 0; }
