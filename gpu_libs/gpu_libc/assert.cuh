#pragma once

#include <cassert>
#include <cstdio>

#define NOT_IMPLEMENTED do { \
    printf("NOT_IMPLEMENTED %s:%d\n", __FILE__, __LINE__); \
    assert(0); \
} while (0) 
