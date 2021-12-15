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

#ifdef GPUMPI_MALLOC_V1
#define GPUMPI_COALESCED
#define malloc __gpu_malloc_v1
#define calloc __gpu_calloc_v1
#define free __gpu_free_v1
#endif
#ifdef GPUMPI_MALLOC_V2
#define GPUMPI_COALESCED
#define malloc __gpu_malloc_v2
#define calloc __gpu_calloc_v2
#define free __gpu_free_v2
#endif
#ifdef GPUMPI_MALLOC_V3
#define GPUMPI_COALESCED
#define malloc __gpu_malloc_v3
#define calloc __gpu_calloc_v3
#define free __gpu_free_v3
#endif
#ifdef GPUMPI_MALLOC_V4
#define GPUMPI_COALESCED
#define malloc __gpu_malloc_v4
#define calloc __gpu_calloc_v4
#define free __gpu_free_v5
#endif
#ifdef GPUMPI_MALLOC_V5
#define GPUMPI_COALESCED
#define malloc __gpu_malloc_v5
#define calloc __gpu_calloc_v5
#define free __gpu_free_v5
#endif
#ifdef GPUMPI_MALLOC_V6
#define GPUMPI_COALESCED
#define malloc __gpu_malloc_coalesce
#define calloc __gpu_calloc_coalesce
#define free __gpu_free_coalesce
#endif

#ifndef GPUMPI_COALESCED
#define malloc __gpu_malloc
#define calloc __gpu_calloc
#define free __gpu_free
#endif

#endif // STDLIB_H_CUH
