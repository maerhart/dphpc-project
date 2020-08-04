#include "unistd.cuh"

#include "assert.cuh"
#include "string.cuh"

__device__ int __gpu_gethostname(char *name, size_t len) {
    const char hostname[] = "GPU thread";
    if (sizeof(hostname) < len) {
        len = sizeof(hostname);
    }
    __gpu_strncpy(name, hostname, len);
    return 0;
}

__device__ int __gpu_getpagesize(void) {
    NOT_IMPLEMENTED;
    return 10;
}

__device__ int __gpu_getopt(int argc, char * const argv[], const char *optstring) {
    NOT_IMPLEMENTED;
    return 0;
}

__device__ unsigned int __gpu_sleep(unsigned int seconds) {
    NOT_IMPLEMENTED;
    return 0;
}

__device__ char *__gpu_optarg;

__device__ int __gpu_optind;
__device__ int __gpu_opterr;
__device__ int __gpu_optopt;
