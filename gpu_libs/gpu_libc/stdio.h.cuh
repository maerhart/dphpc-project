#ifndef GPU_STDIO_H
#define GPU_STDIO_H

#include <stdio.h>

#ifdef FILE
#undef FILE
#endif
#define FILE __GPU_FILE
typedef int FILE;

#ifdef stdout
#undef stdout
#endif
#define stdout __gpu_stdout
__device__ extern FILE *stdout;

#ifdef stderr
#undef stderr
#endif
#define stderr __gpu_stderr
__device__ extern FILE *stderr;

#define fprintf __gpu_fprintf
__device__ int fprintf(FILE * __stream, const char * __format, ...);

#define fflush __gpu_fflush
__device__ int fflush(FILE *__stream);


#define fopen __gpu_fopen
__device__ FILE *fopen(const char *pathname, const char *mode);

#define fclose __gpu_fclose
__device__ int fclose(FILE *stream);

#define fgets __gpu_fgets
__device__ char *fgets(char *s, int size, FILE *stream);

#define putchar __gpu_putchar
__device__ int putchar(int c);

#define sprintf __gpu_sprintf
__device__ int sprintf(char *str, const char *format, ...);


#define puts __gpu_puts
__device__ int puts(const char *s);

#define fgetc __gpu_fgetc
__device__ int fgetc(FILE *stream);

#define feof __gpu_define
__device__ int feof(FILE *stream);

#endif
