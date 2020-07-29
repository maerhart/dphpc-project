#ifndef STRING_H_CUH
#define STRING_H_CUH

#include <string.h>
#include "errno.h.cuh"

#define strcpy __gpu_strcpy
__device__ char *strcpy(char *dest, const char *src);

#define strncpy __gpu_strncpy
__device__ char *strncpy(char *dest, const char *src, size_t n);

#define strcat __gpu_strcat
__device__ char *strcat(char *s, const char *t);

#define strncat __gpu_strncat
__device__ char *strncat(char *s, const char *t);

#define strxfrm __gpu_strxfrm
__device__ size_t strxfrm(char *dest, const char *src, size_t n);

#define strlen __gpu_strlen
__device__ size_t strlen(const char *s);

#define strcmp __gpu_strcmp
__device__ int strcmp(const char *s1, const char *s2);

#define strncmp __gpu_strncmp
__device__ int strncmp(const char *s1, const char *s2, size_t n);

#define strcoll __gpu_strcoll
__device__ int strcoll(const char *s1, const char *s2);

#define strchr __gpu_strchr
__device__ char *strchr(const char *s, int c);

#define strrchr __gpu_strrchr
__device__ char *strrchr(const char *s, int c);

#define strspn __gpu_strspn
__device__ size_t strspn(const char *s, const char *accept);

#define strcspn __gpu_strcspn
__device__ size_t strcspn(const char *s, const char *reject);

#define strpbrk __gpu_strpbrk
__device__ char *strpbrk(const char *s, const char *accept);

#define strstr __gpu_strstr
__device__ char *strstr(const char *haystack, const char *needle);

#define strtok __gpu_strtok
__device__ char *strtok(char *str, const char *delim);

#define strtok_r __gpu_strtok_r
__device__ char *strtok_r(char *str, const char *delim, char** ptrptr);

#define strerror __gpu_strerror
__device__ char* strerror(int errnum);

#define memcpy __gpu_memcpy
__device__ void* memcpy (void *dst, const void *src, size_t n);

#define memccpy __gpu_memccpy
__device__ void *memccpy(void *dst, const void *src, int c, size_t count);

#define memmove __gpu_memmove
__device__ void* memmove(void *dst, const void *src, size_t count);

#define memcmp __gpu_memcmp
__device__ int memcmp(const void *dst, const void *src, size_t count);

#define memchr __gpu_memchr
__device__ void* memchr(const void *s, int c, size_t n);

#define atof __gpu_atof
__device__ double atof(const char *nptr);

#define atoi __gpu_atoi
__device__ int atoi(const char* s);

#define atol __gpu_atol
__device__ long int atol(const char* s);

#define atoll __gpu_atoll
__device__ long long int atoll(const char* s);

#define strtof __gpu_strtof
__device__ float strtof(const char*s , char** endptr);

#define strtod __gpu_strtod
__device__ double strtod(const char* s, char** endptr);

#define strtold __gpu_strtold 
__device__ long double strtold(const char* s, char** endptr);

#define strtoll __gpu_strtoll
__device__ long long int strtoll(const char *nptr, char **endptr, int base);

#define strtol __gpu_strtol
__device__ long int strtol(const char *nptr, char** endptr, int base);

#define strtoull __gpu_strtoull
__device__ unsigned long long int strtoull(const char *ptr, char **endptr, int base);

#define strtoul __gpu_sttoul
__device__ unsigned long long int strtoul(const char *ptr, char **endptr, int base);


#endif // STRING_H_CUH
