#pragma once

#include "stdio.cuh"

#ifdef FILE
#undef FILE
#endif
#define FILE __gpu_FILE

#ifdef stdout
#undef stdout
#endif
#define stdout __gpu_stdout

#ifdef stderr
#undef stderr
#endif
#define stderr __gpu_stderr

#define fprintf __gpu_fprintf
#define fflush __gpu_fflush
#define fopen __gpu_fopen
#define fclose __gpu_fclose
#define fgets __gpu_fgets
#define putchar __gpu_putchar
#define sprintf __gpu_sprintf
#define puts __gpu_puts
#define fgetc __gpu_fgetc
#define feof __gpu_define
