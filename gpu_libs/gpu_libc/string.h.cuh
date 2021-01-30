#ifndef STRING_H_CUH
#define STRING_H_CUH

#include "string.cuh"

#define strcpy __gpu_strcpy
#define strncpy __gpu_strncpy
#define strcat __gpu_strcat
#define strncat __gpu_strncat
#define strxfrm __gpu_strxfrm
#define strlen __gpu_strlen
#define strcmp __gpu_strcmp
#define strncmp __gpu_strncmp
#define strcoll __gpu_strcoll
#define strchr __gpu_strchr
#define strrchr __gpu_strrchr
#define strspn __gpu_strspn
#define strcspn __gpu_strcspn
#define strpbrk __gpu_strpbrk
#define strstr __gpu_strstr
#define strtok __gpu_strtok
#define strtok_r __gpu_strtok_r
#define strerror __gpu_strerror
#define memcpy __gpu_memcpy
#define memccpy __gpu_memccpy
#define memmove __gpu_memmove
#define memcmp __gpu_memcmp
#define memchr __gpu_memchr
#define atof __gpu_atof
#define atoi __gpu_atoi
#define atol __gpu_atol
#define atoll __gpu_atoll
#define strtof __gpu_strtof
#define strtod __gpu_strtod
#define strtold __gpu_strtold 
#define strtoll __gpu_strtoll
#define strtol __gpu_strtol
#define strtoull __gpu_strtoull
#define strtoul __gpu_sttoul
#define strdup __gpu_strdup

#endif // STRING_H_CUH
