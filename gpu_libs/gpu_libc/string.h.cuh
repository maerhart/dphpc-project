#ifndef STRING_H_CUH
#define STRING_H_CUH

#include <string.h>

#define strcmp __gpu_strcmp
__device__ int strcmp(const char *s1, const char *s2);

#define strncmp __gpu_strncmp
__device__ int strncmp(const char *s1, const char *s2, size_t n);

#define strcpy __gpu_strcpy
__device__ char *strcpy(char *dest, const char *src);

#define strncpy __gpu_strncpy
__device__ char *strncpy(char *dest, const char *src, size_t n);

#define strstr __gpu_strstr
__device__ char *strstr(const char *haystack, const char *needle);

#define strtok __gpu_strtok
__device__ char *strtok(char *str, const char *delim);

#define strlen __gpu_strlen
__device__ size_t strlen(const char *s);

#define strcat __gpu_strcat
__device__ char *strcat(char *dest, const char *src);

#define strdup __gpu_strdup
__device__ char *strdup(const char *s);

#define strchr __gpu_strchr
__device__ char *strchr(const char *s, int c);

#endif // STRING_H_CUH
