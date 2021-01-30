#pragma once

#include <string.h>
#include "errno.cuh"

__device__ char *__gpu_strcpy(char *dest, const char *src);
__device__ char *__gpu_strncpy(char *dest, const char *src, size_t n);
__device__ char *__gpu_strcat(char *s, const char *t);
__device__ char *__gpu_strncat(char *s, const char *t);
__device__ size_t __gpu_strxfrm(char *dest, const char *src, size_t n);
__device__ size_t __gpu_strlen(const char *s);
__device__ int __gpu_strcmp(const char *s1, const char *s2);
__device__ int __gpu_strncmp(const char *s1, const char *s2, size_t n);
__device__ int __gpu_strcoll(const char *s1, const char *s2);
__device__ char *__gpu_strchr(const char *s, int c);
__device__ char *__gpu_strrchr(const char *s, int c);
__device__ size_t __gpu_strspn(const char *s, const char *accept);
__device__ size_t __gpu_strcspn(const char *s, const char *reject);
__device__ char *__gpu_strpbrk(const char *s, const char *accept);
__device__ char *__gpu_strstr(const char *haystack, const char *needle);
__device__ char *__gpu_strtok(char *str, const char *delim);
__device__ char *__gpu_strtok_r(char *str, const char *delim, char** ptrptr);
__device__ char* __gpu_strerror(int errnum);
__device__ void* __gpu_memcpy(void *dst, const void *src, size_t n);
__device__ void *__gpu_memccpy(void *dst, const void *src, int c, size_t count);
__device__ void* __gpu_memmove(void *dst, const void *src, size_t count);
__device__ int __gpu_memcmp(const void *dst, const void *src, size_t count);
__device__ void* __gpu_memchr(const void *s, int c, size_t n);
__device__ double __gpu_atof(const char *nptr);
__device__ int __gpu_atoi(const char* s);
__device__ long int __gpu_atol(const char* s);
__device__ long long int __gpu_atoll(const char* s);
__device__ float __gpu_strtof(const char*s , char** endptr);
__device__ double __gpu_strtod(const char* s, char** endptr);
__device__ long double __gpu_strtold(const char* s, char** endptr);
__device__ long long int __gpu_strtoll(const char *nptr, char **endptr, int base);
__device__ long int __gpu_strtol(const char *nptr, char** endptr, int base);
__device__ unsigned long long int __gpu_strtoull(const char *ptr, char **endptr, int base);
__device__ unsigned long int __gpu_strtoul(const char *ptr, char **endptr, int base);
__device__ char *__gpu_strdup(const char *s);