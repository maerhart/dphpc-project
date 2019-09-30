#include "string.h.cuh"

__device__ int strcmp(const char *s1, const char *s2) {
    return 0;
}

__device__ int strncmp(const char *s1, const char *s2, size_t n) {
    return 0;
}

__device__ char *strcpy(char *dest, const char *src) {
    return dest;
}

__device__ static char stub[] = "stub";

__device__ char *strstr(const char *haystack, const char *needle) {
    return stub;
}

__device__ char *strtok(char *str, const char *delim) {
    return str;
}

__device__ size_t strlen(const char *s) {
    return 0;
}

__device__ char *strcat(char *dest, const char *src) {
    return dest;
}

__device__ char *strdup(const char *s) {
    return stub;
}

__device__ char *strchr(const char *s, int c) {
    return stub;
}
