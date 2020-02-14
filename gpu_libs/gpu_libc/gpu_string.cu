#include "string.h.cuh"

#include "assert.h.cuh"

__device__ int strcmp(const char *s1, const char *s2) {
    while (*s1 != '\0' && *s1 == *s2) {
        s1++;
        s2++;
    }
    return s1 - s2;
}

__device__ int strncmp(const char *s1, const char *s2, size_t n) {
    NOT_IMPLEMENTED
    return 0;
}

__device__ char *strcpy(char *dest, const char *src) {
    for (size_t i = 0; src[i] != '\0'; i++) {
        dest[i] = src[i];
    }
    return dest;
}

__device__ char *strncpy(char *dest, const char *src, size_t n) {
    size_t i = 0;
    while (i < n && src[i] != '\0') {
        dest[i] = src[i];
        i++;
    }
    while (i < n) {
        dest[i] = '\0';
        i++;
    }
    return dest;
}

__device__ static char stub[] = "stub";

__device__ char *strstr(const char *haystack, const char *needle) {
    NOT_IMPLEMENTED
    return stub;
}

__device__ char *strtok(char *str, const char *delim) {
    NOT_IMPLEMENTED
    return str;
}

__device__ size_t strlen(const char *s) {
    NOT_IMPLEMENTED
    return 0;
}

__device__ char *strcat(char *dest, const char *src) {
    NOT_IMPLEMENTED
    return dest;
}

__device__ char *strdup(const char *s) {
    NOT_IMPLEMENTED
    return stub;
}

__device__ char *strchr(const char *s, int c) {
    NOT_IMPLEMENTED
    return stub;
}
