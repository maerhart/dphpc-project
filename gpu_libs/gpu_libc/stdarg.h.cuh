#ifndef STDARG_H_CUH
#define STDARG_H_CUH

#include <stdarg.h>

#define vfprintf __gpu_vfprintf
__device__ int vfprintf(FILE *stream, const char *format, va_list ap);

#endif // STDARG_H_CUH
