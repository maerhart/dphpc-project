#ifndef DIRENT_H_CUH
#define DIRENT_H_CUH

#include <dirent.h>

#define DIR __GPU_DIR
typedef struct DIR_t {} DIR;

#define dirent __gpu_dirent
struct dirent {
     char d_name[256];
};

#define opendir __gpu_opendir
__device__ DIR *opendir(const char *name);

#define readdir __gpu_readdir
__device__ struct dirent *readdir(DIR *dirp);

#define closedir __gpu_closedir
__device__ int closedir(DIR *dirp);

#endif // DIRENT_H_CUH
