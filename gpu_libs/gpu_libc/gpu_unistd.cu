#include "unistd.h.cuh"

__device__ int gethostname(char *name, size_t len) {
    return 0;
}

__device__ int getpagesize(void) {
    return 10;
}

__device__ int getopt(int argc, char * const argv[],
                      const char *optstring) {
    return 0;
}

__device__ char *optarg;

__device__ int optind, opterr, optopt;

__device__ unsigned int sleep(unsigned int seconds) {
    return 0;
}
