#pragma once

#include "ctype.cuh"

#define isalnum __gpu_isalnum
#define isalpha __gpu_isalpha
#define islower __gpu_islower
#define isupper __gpu_isupper
#define isdigit  __gpu_isdigit
#define isxdigit __gpu_isxdigit
#define iscntrl  __gpu_iscntrl
#define isgraph __gpu_isgraph
#define isspace __gpu_isspace
#define isblank __gpu_isblank
#define isprint __gpu_isprint
#define ispunct __gpu_ispunct
#define tolower __gpu_tolower
#define toupper __gpu_toupper


