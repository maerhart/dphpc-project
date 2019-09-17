#ifndef GPU_STDIO_H
#define GPU_STDIO_H

#include <stdio.h>

typedef struct FILE_t {} __GPU_FILE;
#ifdef FILE
#undef FILE
#endif
#define FILE __GPU_FILE

__device__ extern FILE *__gpu_stdout;
#ifdef stdout
#undef stdout
#endif
#define stdout __gpu_stdout

__device__ int fprintf (FILE * __stream, const char * __format, ...);

__device__ int fflush (FILE *__stream);

#endif
