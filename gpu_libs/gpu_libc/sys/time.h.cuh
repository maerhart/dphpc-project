#ifndef TIME_H_CUH
#define TIME_H_CUH

#include "sys/time.h"

#define gettimeofday __gpu_gettimeofday
__device__ int gettimeofday(struct timeval *tv, struct timezone *tz);

#define localtime __gpu_localtime
__device__ struct tm *localtime(const time_t *timep);

#define time __gpu_gettimeofday
__device__ time_t time(time_t *tloc);

#define strftime __gpu_strftime
__device__ size_t strftime(char *s, size_t max, const char *format,
                       const struct tm *tm);

#endif // TIME_H_CUH
