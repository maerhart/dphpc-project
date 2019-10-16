#include "unistd.h.cuh"

#include "assert.h.cuh"

__device__ int gethostname(char *name, size_t len) {
    NOT_IMPLEMENTED
    return 0;
}

__device__ int getpagesize(void) {
    NOT_IMPLEMENTED
    return 10;
}

__device__ int getopt(int argc, char * const argv[], const char *optstring) {
    NOT_IMPLEMENTED
    return 0;
}

__device__ unsigned int sleep(unsigned int seconds) {
    NOT_IMPLEMENTED
    return 0;
}

__device__ char *optarg;

__device__ int optind, opterr, optopt;
