#pragma once

#include <sys/time.h>

__device__ int __gpu_gettimeofday(struct timeval *tv, struct timezone *tz);
__device__ struct tm *__gpu_localtime(const time_t *timep);
__device__ time_t __gpu_time(time_t *tloc);
__device__ size_t __gpu_strftime(char *s, size_t max, const char *format,
                       const struct tm *tm);