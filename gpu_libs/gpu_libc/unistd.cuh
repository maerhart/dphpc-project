#pragma once

#include <unistd.h>

#include "getopt.cuh"

__device__ int __gpu_gethostname(char *name, size_t len);

__device__ int __gpu_getpagesize(void);

__device__ int __gpu_getopt(int argc, char * const argv[], const char *optstring);

__device__ extern char *__gpu_optarg;

__device__ extern int __gpu_optind;
__device__ extern int __gpu_opterr;
__device__ extern int __gpu_optopt;

__device__ unsigned int __gpu_sleep(unsigned int seconds);

