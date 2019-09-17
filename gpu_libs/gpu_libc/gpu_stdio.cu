#include "gpu_stdio.h"


__device__ int fprintf (FILE * __stream, const char * __format, ...) { return 0; }

__device__ int fflush (FILE *__stream) { return 0; }

__device__ FILE *stdout;
