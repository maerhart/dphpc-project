#pragma once

#include <stdarg.h>

#include "stdio.cuh"

__device__ int __gpu_vfprintf(__gpu_FILE *stream, const char *format, va_list ap);
