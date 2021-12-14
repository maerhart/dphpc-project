#include <iostream>
#include "dynamic_allocator.cu"
#include "warp_malloc.cu"
#include "../gpu_libs/gpu_malloc/dyn_malloc.cu"

#define COALESCE true

#define VERBOSE true
#define SIZE 37*sizeof(float)

__global__ void baseline(uintptr_t *malloc_locations) {
    int id = (blockIdx.x*blockDim.x + threadIdx.x);
    void* ptr = (void*) malloc_baseline(SIZE, COALESCE);
    malloc_locations[id] = (uintptr_t)ptr;
    
    if (VERBOSE) {
        int align = 32;
        printf("id: %d, %p, %d-byte-aligned: %s\n", id, ptr, align, ((uint)((uintptr_t)ptr % align == 0) ? "true":"false"));
    }
        
    free_baseline(ptr);
}

__global__ void v1_flo(uintptr_t *malloc_locations) {
    int id = (blockIdx.x*blockDim.x + threadIdx.x);
    void* ptr = (void*) malloc_v1(SIZE, COALESCE);
    malloc_locations[id] = (uintptr_t)ptr;
    
    //free_v1(ptr);
}

__global__ void v1_martin(uintptr_t *malloc_locations) {
    int id = (blockIdx.x*blockDim.x + threadIdx.x);
    void* ptr = (void*) dyn_malloc(SIZE, COALESCE);
    malloc_locations[id] = (uintptr_t)ptr;
    
    dyn_free(ptr);
}
/*
__global__ void v3_nils(uintptr_t *malloc_locations) {
    init_malloc_v3();
    int id = (blockIdx.x*blockDim.x + threadIdx.x);
    void* ptr = (void*) malloc_v3(SIZE, COALESCE);
    malloc_locations[id] = (uintptr_t)ptr;
    
    free_v3(ptr);
    clean_malloc_v3();
}
*/
__global__ void v4_warp_anton(uintptr_t *malloc_locations) {
    int id = (blockIdx.x*blockDim.x + threadIdx.x);
    void* ptr = (void*) malloc_v4(SIZE, COALESCE);
    malloc_locations[id] = (uintptr_t)ptr;
    
    free_v4(ptr);
}

void run_test(const std::string& name, int blocks, int threads_per_block, void(*kernel)(uintptr_t*)) {
    std::cout << "Running " << name << " ...  " << std::endl;

    int total_threads = blocks * threads_per_block;
    
    uintptr_t malloc_locations[total_threads];
    uintptr_t *d_malloc_locations;
    cudaMalloc(&d_malloc_locations, total_threads*sizeof(uintptr_t));
    
    kernel<<<blocks, threads_per_block>>>(d_malloc_locations);
    cudaDeviceSynchronize(); // to allow for printf in kernel code

    cudaMemcpy(malloc_locations, d_malloc_locations, total_threads*sizeof(uintptr_t), cudaMemcpyDeviceToHost);
    
    // check for error
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
       std::cout << "CUDA ERROR: " << cudaGetErrorString(err) << std::endl;
       exit(-1);
    }
    
    // check alignment
    bool all_alignment[6] = {true};
    for (int id = 0; id < total_threads; ++id) {
        uintptr_t ptr = malloc_locations[id];

        all_alignment[0] &= ptr % 4 == 0;
        all_alignment[1] &= ptr % 8 == 0;
        all_alignment[2] &= ptr % 16 == 0;
        all_alignment[3] &= ptr % 32 == 0;
        all_alignment[4] &= ptr % 64 == 0;
        all_alignment[5] &= ptr % 128 == 0;
    }
    
    printf("  all alignment:\n");
    printf("    4-byte-aligned: %s\n", all_alignment[0] ? "true":"false");
    printf("    8-byte-aligned: %s\n", all_alignment[1] ? "true":"false");
    printf("    16-byte-aligned: %s\n", all_alignment[2] ? "true":"false");
    printf("    32-byte-aligned: %s\n", all_alignment[3] ? "true":"false");
    printf("    64-byte-aligned: %s\n", all_alignment[4] ? "true":"false");
    printf("    128-byte-aligned: %s\n", all_alignment[5] ? "true":"false");
    
    
    bool block_alignment[6] = {true};
    for (int b = 0; b < blocks; ++b) {
        int id = b*threads_per_block;
        uintptr_t ptr = malloc_locations[id];

        block_alignment[0] &= ptr % 4 == 0;
        block_alignment[1] &= ptr % 8 == 0;
        block_alignment[2] &= ptr % 16 == 0;
        block_alignment[3] &= ptr % 32 == 0;
        block_alignment[4] &= ptr % 64 == 0;
        block_alignment[5] &= ptr % 128 == 0;
    }
    
    printf("  block alignment:\n");
    printf("    4-byte-aligned: %s\n", block_alignment[0] ? "true":"false");
    printf("    8-byte-aligned: %s\n", block_alignment[1] ? "true":"false");
    printf("    16-byte-aligned: %s\n", block_alignment[2] ? "true":"false");
    printf("    32-byte-aligned: %s\n", block_alignment[3] ? "true":"false");
    printf("    64-byte-aligned: %s\n", block_alignment[4] ? "true":"false");
    printf("    128-byte-aligned: %s\n", block_alignment[5] ? "true":"false");
    
    
    if (VERBOSE) {
        for (int b = 0; b < blocks; ++b) {
            printf("  block %d\n", b);
            for (int t = 0; t < threads_per_block; ++t) {
                int id = b*threads_per_block + t;
                if (t == 0) {
                    uintptr_t ptr = malloc_locations[id];
                    printf("    thread %d: address %p\n", t, (void*)ptr);
                } else {
                    int offset = malloc_locations[id] - malloc_locations[b*threads_per_block];
                    int dist = malloc_locations[id] - malloc_locations[id-1];
                    printf("    thread %d: offset to first thread: %d, distance to previous: %d\n", t, offset, dist);
                }
            }
        }
    }
    
}

// test alignment of different mallocs
int main(int argc, char **argv) {
    // read arguments
	int blocks = atoi(argv[1]);
	int threads_per_block = atoi(argv[2]);
    
    run_test("baseline", blocks, threads_per_block, baseline);
    run_test("v1_flo", blocks, threads_per_block, v1_flo);
    run_test("v1_martin", blocks, threads_per_block, v1_martin);
    //run_test("v3_nils", blocks, threads_per_block, v3_nils);
    //run_test("v4_warp_anton", blocks, threads_per_block, v4_warp_anton);
    return 0;
}