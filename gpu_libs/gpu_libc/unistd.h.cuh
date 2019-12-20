#ifndef UNISTD_H_CUH
#define UNISTD_H_CUH

#include <unistd.h>

#include "getopt.h.cuh"

#define gethostname __gpu_gethostname
__device__ int gethostname(char *name, size_t len);

#define getpagesize __gpu_getpagesize
__device__ int getpagesize(void);

#define getopt __gpu_getopt
__device__ int getopt(int argc, char * const argv[],
                  const char *optstring);

#define optarg __gpu_optarg
__device__ extern char *optarg;

#define optind __gpu_optind
#define opterr __gpu_opterr
#define optopt __gpu_optopt
__device__ extern int optind, opterr, optopt;

#define sleep __gpu_sleep
__device__ unsigned int sleep(unsigned int seconds);

#endif // UNISTD_H_CUH
