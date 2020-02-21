#ifndef DEVICE_HOST_COMM_H
#define DEVICE_HOST_COMM_H

__device__ void* allocate_host_mem(size_t size);

__device__ void free_host_mem(void* ptr);

__device__ void delegate_to_host(void* mem, size_t size);

#endif
