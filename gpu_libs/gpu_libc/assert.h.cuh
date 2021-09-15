#pragma once

#include "assert.cuh"

#ifdef assert
    #undef assert
#endif

#define assert __gpu_assert
