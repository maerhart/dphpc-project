#pragma once

#include <cctype>

__device__ int __gpu_isalnum(int ch);

__device__ int __gpu_isalpha(int ch);

__device__ int __gpu_islower( int ch );

__device__ int __gpu_isupper(int c);

__device__ int __gpu_isdigit( int ch );

__device__ int __gpu_isxdigit( int ch );

__device__ int __gpu_iscntrl( int ch );

__device__ int __gpu_isgraph(int x);

__device__ int __gpu_isspace( int ch );

__device__ int __gpu_isblank(int ch);

__device__ int __gpu_isprint(int x);

__device__ int __gpu_ispunct( int ch );

__device__ int __gpu_tolower(int ch);

__device__ int __gpu_toupper(int ch);
