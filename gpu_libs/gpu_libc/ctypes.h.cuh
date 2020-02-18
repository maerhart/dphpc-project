#ifndef CTYPES_H_CUH
#define CTYPES_H_CUH

#define isalnum __gpu_isalnum
__device__ int isalnum(int ch);

#define isalpha __gpu_isalpha
__device__ int isalpha(int ch);

#define islower __gpu_islower
__device__ int islower( int ch );

#define isupper __gpu_isupper
__device__ int isupper(int c);

#define isdigit  __gpu_isdigit
__device__ int isdigit ( int ch );

#define isxdigit __gpu_isxdigit
__device__ int isxdigit( int ch );

#define iscntrl  __gpu_iscntrl
__device__ int iscntrl( int ch );

#define isgraph __gpu_isgraph
__device__ int isgraph(int x);

#define isspace __gpu_isspace
__device__ int isspace( int ch );

#define isblank __gpu_isblank
__device__ int isblank(int ch);

#define isprint __gpu_isprint
__device__ int isprint(int x);

#define ispunct __gpu_ispunct
__device__ int ispunct( int ch );

#define tolower __gpu_tolower
__device__ int tolower(int ch);

#define toupper __gpu_toupper
__device__ int toupper(int ch);

#endif 
