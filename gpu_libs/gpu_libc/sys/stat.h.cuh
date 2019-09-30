#ifndef STAT_H_CUH
#define STAT_H_CUH

#include <sys/stat.h>

#define dev_t int

#define stat __gpu_stat
struct stat {
    dev_t st_rdev;
};

#define stat __gpu_stat
__device__ int stat(const char *pathname, struct stat *statbuf);

#endif // STAT_H_CUH
