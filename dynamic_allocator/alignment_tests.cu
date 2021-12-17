#include <iostream>
#include <algorithm>
#include "../gpu_libs/gpu_malloc/dynamic_allocator.cu"
#include "../gpu_libs/gpu_malloc/warp_malloc.cu"
#include "../gpu_libs/gpu_malloc/dyn_malloc.cu"

#define COALESCE true

#define VERBOSE false
#define SIZE 20*sizeof(float)

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
    
    free_v1(ptr);
}

__global__ void v1_flo_per_warp_no_headers(uintptr_t *malloc_locations) {
    int id = (blockIdx.x*blockDim.x + threadIdx.x);
    void* ptr = (void*) malloc_v1_per_warp_no_headers(SIZE, COALESCE);
    malloc_locations[id] = (uintptr_t)ptr;
    
    free_v1_per_warp_no_headers(ptr);
}

__global__ void v1_flo_per_block_no_headers_warp_align(uintptr_t *malloc_locations) {
    int id = (blockIdx.x*blockDim.x + threadIdx.x);
    void* ptr = (void*) malloc_v1_per_block_no_headers_warp_align(SIZE, COALESCE);
    malloc_locations[id] = (uintptr_t)ptr;
    
    free_v1_per_block_no_headers_warp_align(ptr);
}

__global__ void v1_flo_per_block_no_headers_warp_align_dist(uintptr_t *malloc_locations) {
    int id = (blockIdx.x*blockDim.x + threadIdx.x);
    void* ptr = (void*) malloc_v1_per_block_no_headers_warp_align_dist(SIZE, COALESCE);
    malloc_locations[id] = (uintptr_t)ptr;
    
    free_v1_per_block_no_headers_warp_align_dist(ptr);
}

__global__ void v1_flo_per_block_no_headers_warp_align_80(uintptr_t *malloc_locations) {
    int id = (blockIdx.x*blockDim.x + threadIdx.x);
    void* ptr = (void*) malloc_v1_per_block_no_headers_warp_align_80(SIZE, COALESCE, 13);
    malloc_locations[id] = (uintptr_t)ptr;
    
    free_v1_per_block_no_headers_warp_align_80(ptr);
}

__global__ void v1_martin(uintptr_t *malloc_locations) {
    int id = (blockIdx.x*blockDim.x + threadIdx.x);
    void* ptr = (void*) dyn_malloc(SIZE, COALESCE);
    malloc_locations[id] = (uintptr_t)ptr;
    
    dyn_free(ptr);
}

__global__ void v3_nils(uintptr_t *malloc_locations) {
    init_malloc_v3();
    int id = (blockIdx.x*blockDim.x + threadIdx.x);
    void* ptr = (void*) malloc_v3(SIZE, COALESCE);
    malloc_locations[id] = (uintptr_t)ptr;
    
    free_v3(ptr);
    clean_malloc_v3();
}

__global__ void v4_warp_anton(uintptr_t *malloc_locations) {
    int id = (blockIdx.x*blockDim.x + threadIdx.x);
    void* ptr = (void*) malloc_v4(SIZE, COALESCE);
    malloc_locations[id] = (uintptr_t)ptr;
    
    free_v4(ptr);
}

__global__ void v5_warp_anton(uintptr_t *malloc_locations) {
    int id = (blockIdx.x*blockDim.x + threadIdx.x);
    void* ptr = (void*) malloc_v5(SIZE);
    malloc_locations[id] = (uintptr_t)ptr;
    
    free_v5(ptr);
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
    bool thread_alignment[6] = {true, true, true, true, true, true};
    for (int id = 0; id < total_threads; ++id) {
        uintptr_t ptr = malloc_locations[id];

        thread_alignment[0] &= ptr % 4 == 0;
        thread_alignment[1] &= ptr % 8 == 0;
        thread_alignment[2] &= ptr % 16 == 0;
        thread_alignment[3] &= ptr % 32 == 0;
        thread_alignment[4] &= ptr % 64 == 0;
        thread_alignment[5] &= ptr % 128 == 0;
    }
    int i = 0;
    int power_of_2 = 2;
    while (thread_alignment[i] & i < 6) {
        i++;
        power_of_2 *= 2;
    }
    
    printf("  thread (abs) alignment: %d-byte-aligned\n", power_of_2);
    
    bool block_alignment[6] = {true, true, true, true, true, true};
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
    
    i = 0;
    power_of_2 = 2;
    while (block_alignment[i] & i < 6) {
        i++;
        power_of_2 *= 2;
    }
    
    printf("  block (abs) alignment:  %d-byte-aligned\n", power_of_2);
    
    bool warp_alignment[6] = {true, true, true, true, true, true};
    int warps = total_threads / 32;
    for (int w = 0; w < warps; ++w) {
        int id = w*32;
        uintptr_t ptr = malloc_locations[id];

        warp_alignment[0] &= ptr % 4 == 0;
        warp_alignment[1] &= ptr % 8 == 0;
        warp_alignment[2] &= ptr % 16 == 0;
        warp_alignment[3] &= ptr % 32 == 0;
        warp_alignment[4] &= ptr % 64 == 0;
        warp_alignment[5] &= ptr % 128 == 0;
    }
    
    i = 0;
    power_of_2 = 2;
    while (warp_alignment[i] & i < 6) {
        i++;
        power_of_2 *= 2;
    }
    
    printf("  warp (abs) alignment:   %d-byte-aligned\n", power_of_2);
    
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
    
    
    // print spacing of warps
    for (int b = 0; b < blocks; ++b) {
        printf("  block %d\n", b);
        for (int t = 0; t < threads_per_block; t+=32) {
            int id = b*threads_per_block + t;
            
            uintptr_t ptr = malloc_locations[id];
            printf("    warp %d: address %p\n", t/32, (void*)ptr);
            
        }
    }
    
    int num_warps = total_threads / 32;
    uintptr_t *warp_starts = (uintptr_t*) malloc(num_warps * sizeof(uintptr_t));
    for (int i = 0; i < num_warps; ++i) {
        warp_starts[i] = malloc_locations[i*32];
        //printf("    warp %d: address %p\n", i, (void*)warp_starts[i]);
    }
    std::sort(warp_starts, warp_starts + num_warps);
    for (int i = 1; i < num_warps; ++i) {
        //printf("    warp %d: address %p\n", i, (void*)warp_starts[i]);
        int dist = warp_starts[i] - warp_starts[i-1];
        printf("    warp %d: address %p, distance to previous: %d\n", i, (void*)warp_starts[i], dist);
    }
    
}

// test alignment of different mallocs
int main(int argc, char **argv) {
    // read arguments
	int blocks = atoi(argv[1]);
	int threads_per_block = atoi(argv[2]);
    
    
    printf("Allocation size: %lu bytes\n", SIZE);
    
    run_test("baseline", blocks, threads_per_block, baseline);
    //run_test("v1_flo", blocks, threads_per_block, v1_flo);
    //run_test("v1_flo_per_warp_no_headers", blocks, threads_per_block, v1_flo_per_warp_no_headers);
    
    //run_test("v1_flo_per_block_no_headers_warp_align", blocks, threads_per_block, v1_flo_per_block_no_headers_warp_align);
    //run_test("v1_flo_per_block_no_headers_warp_align_dist", blocks, threads_per_block, v1_flo_per_block_no_headers_warp_align_dist);
    run_test("v1_flo_per_block_no_headers_warp_align_80", blocks, threads_per_block, v1_flo_per_block_no_headers_warp_align_80);
    
    //run_test("v1_martin", blocks, threads_per_block, v1_martin);
    //run_test("v3_nils", blocks, threads_per_block, v3_nils);
    //run_test("v4_warp_anton", blocks, threads_per_block, v4_warp_anton);
    //run_test("v5_warp_anton", blocks, threads_per_block, v5_warp_anton);
    return 0;
}