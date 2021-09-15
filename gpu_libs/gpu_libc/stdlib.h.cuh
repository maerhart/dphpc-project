#ifndef STDLIB_H_CUH
#define STDLIB_H_CUH

#include "stdlib.cuh"

#ifdef RAND_MAX
#undef RAND_MAX
#endif
#define RAND_MAX __gpu_RAND_MAX

#define qsort __gpu_qsort
#define atoi __gpu_atoi
#define atol __gpu_atol
#define realloc __gpu_realloc
#define srand __gpu_srand
#define rand __gpu_rand
#define getenv __gpu_getenv
#define exit __gpu_exit
#define abort __gpu_abort
#define strtod __gpu_strtod
#define posix_memalign __gpu_posix_memalign
#define malloc __gpu_malloc
#define calloc __gpu_calloc

#endif // STDLIB_H_CUH
