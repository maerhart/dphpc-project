#pragma once

#include "time.cuh"

#define gettimeofday __gpu_gettimeofday
#define localtime __gpu_localtime
#define time __gpu_gettimeofday
#define strftime __gpu_strftime