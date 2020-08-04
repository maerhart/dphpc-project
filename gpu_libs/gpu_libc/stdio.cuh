#pragma once

#include <cstdio>

typedef int __gpu_FILE;

__device__ extern __gpu_FILE *__gpu_stdout;

__device__ extern __gpu_FILE *__gpu_stderr;

__device__ int __gpu_fprintf(__gpu_FILE * __stream, const char * __format, ...);

__device__ int __gpu_fflush(__gpu_FILE *__stream);

__device__ __gpu_FILE *__gpu_fopen(const char *pathname, const char *mode);

__device__ int __gpu_fclose(__gpu_FILE *stream);

__device__ char *__gpu_fgets(char *s, int size, __gpu_FILE *stream);

__device__ int __gpu_putchar(int c);

__device__ int __gpu_sprintf(char *str, const char *format, ...);

__device__ int __gpu_puts(const char *s);

__device__ int __gpu_fgetc(__gpu_FILE *stream);

__device__ int __gpu_feof(__gpu_FILE *stream);

