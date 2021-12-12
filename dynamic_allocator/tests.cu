#include <iostream>
#include "dynamic_allocator.cu"
//#include "warp_malloc.cu"
#include "warp_malloc_v2.cu"

#define MALLOC malloc_v5
#define FREE free_v5

// allocate one int per thread and set to threadId
__global__ void test(int *resulting_ids) {
    int id = (blockIdx.x*blockDim.x + threadIdx.x);
    int* val = (int*) MALLOC(sizeof(int));
    *val = id;
    resulting_ids[id] = *val;
    FREE(val);
}

__global__ void test_different_size(int *resulting_ids) {
    int id = (blockIdx.x*blockDim.x + threadIdx.x);
    size_t size = sizeof(int) * (1 + (id % 32));
    int* val = (int*) MALLOC(size);
    val[id % 32] = id;  // write to very end of segment
    resulting_ids[id] = val[id % 32];
    FREE(val);
}

__global__ void test_different_types(int *resulting_ids) {
    // good for alignemnt requirements
    int id = (blockIdx.x*blockDim.x + threadIdx.x);
    int choice = id % 5;
    if (choice == 0 && ((char) id) == id) {
        char* val = (char*) MALLOC(sizeof(char));
        *val = id;
        resulting_ids[id] = *val;
        FREE(val);
    } else if (choice == 1 && ((short int) id) == id) {
        short int* val = (short int*) MALLOC(sizeof(short int));
        *val = id;
        resulting_ids[id] = *val;
        FREE(val);
    } else if (choice == 2) {
        long int* val = (long int*) MALLOC(sizeof(long int)); // size 64 bits
        *val = id;
        resulting_ids[id] = *val;
        FREE(val);
    } else if (choice == 3) { // TODO do we need this?
    	// check for 128 bits
        int* val = (int*) MALLOC(16);
    	assert(((long) val) % 16 == 0);
        *val = id;
        resulting_ids[id] = *val;
        FREE(val);
    } else if (choice == 456) { // TODO normal malloc wouldn't even pass this. why?
        // check that alignment correct for max_align_t
        int max_size = sizeof(max_align_t);
        int* val = (int*) MALLOC(max_size);
    	assert(((long) val) % max_size == 0);
        *val = id;
        resulting_ids[id] = *val;
        FREE(val);
    } else {
        int* val = (int*) MALLOC(sizeof(int));
        *val = id;
        resulting_ids[id] = *val;
        FREE(val);
    }
}

/**
 * Passes pointers around and doesn't free all at same time
 */
__global__ void test_pass_ptrs(int *resulting_ids) {

    __shared__ int** ptrs;
    // let thread 0 in block allocate array for entire block
    if (threadIdx.x == 0) {
        ptrs = (int**) MALLOC(sizeof(int*) * blockDim.x);
    } 
    __syncthreads();

    // malloc int and share pointer
    int id = (blockIdx.x*blockDim.x + threadIdx.x);
    int* val = (int*) MALLOC(sizeof(int));
    assert(val != NULL);
    ptrs[threadIdx.x] = val;
    __syncthreads();

    // NOTE lane and warp get modified for the shuffling of values
    int lane = threadIdx.x % 32;
    int warp = threadIdx.x / 32;
    bool even_thread = lane % 2 == 0 && warp % 2 == 0;

    // write id if everything even
    if (even_thread) {
        *val = id;
    }
    __syncthreads(); // make sure that pointer written before read
    if (even_thread) {
        // shuffle pointers around warps

        // map lane i warp j  -> warp i lane j
        // if i >= #warps or j >= #lanes => leave

        if (lane < blockDim.x / 32 && warp < 32) {
            int temp = lane;
            lane = warp;
            warp = temp;
        }
        
        val = ptrs[warp * 32 + lane];
        resulting_ids[id] = *val;
        FREE(val);
    }

    // do the same for odd threads
    if (!even_thread) {
        *val = id;
    }
    __syncthreads();
    if (!even_thread) {
        if (lane < blockDim.x / 32 && warp < 32) {
            int temp = lane;
            lane = warp;
            warp = temp;
        }
        val = ptrs[warp * 32 + lane];
        resulting_ids[id] = *val;
        FREE(val);
    }

    __syncthreads();
    if (threadIdx.x == 0) {
        FREE(ptrs);
    }
}


void run_test(const std::string& name, int blocks, int threads_per_block, void(*kernel)(int*)) {
    std::cout << "Running " << name << " ...  ";

    int total_threads = blocks * threads_per_block;

    int resulting_ids[total_threads];
    int *d_resulting_ids;
    cudaMalloc(&d_resulting_ids, total_threads*sizeof(int));

    kernel<<<blocks, threads_per_block>>>(d_resulting_ids);
    cudaDeviceSynchronize(); // to allow for printf in kernel code

    // check for error
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
       std::cout << "CUDA ERROR: " << cudaGetErrorString(err) << std::endl;
       exit(-1);
    }

    cudaMemcpy(resulting_ids, d_resulting_ids, total_threads*sizeof(int), cudaMemcpyDeviceToHost);

    // sum up all ids, should match to the sum from 0 to total_threads
    bool passed = true;
    int sum = 0;
    for (int i = 0; i < total_threads; ++i) {
        sum += resulting_ids[i];
    }
    // sum up 0 to total_threads
    passed = sum == (total_threads - 1) * (total_threads) / 2;

    if (passed) {
        std::cout << "Tests passed" << std::endl;
    }
    else {
        std::cout << "Failed" << std::endl;
    }
    
}


int main(int argc, char* argv[]) {
    // run some simple unit tests, only in debug mode!
    int blocks = 100;
    int threads_per_block = 128;
    run_test("basic          ", blocks, threads_per_block, test);
    run_test("different sizes", blocks, threads_per_block, test_different_size);
    run_test("different types", blocks, threads_per_block, test_different_types);
    run_test("pass ptrs      ", blocks, threads_per_block, test_pass_ptrs);

    return 0;
}
