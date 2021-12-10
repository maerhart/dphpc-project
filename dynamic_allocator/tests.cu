#include <iostream>
#include "dynamic_allocator.cu"
#include "warp_malloc.cu"

#define MALLOC malloc_v4
#define FREE free_v4

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
    val[id % 32] = id;  // write to very end of sement
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
        long int* val = (long int*) MALLOC(sizeof(long int));
        *val = id;
        resulting_ids[id] = *val;
        FREE(val);
    } else if (choice == 3) {
        long long int* val = (long long int*) MALLOC(sizeof(long long int));
        *val = id;
        resulting_ids[id] = *val;
        FREE(val);
    } else if (choice == 1234) {
	    /* TODO
	// check that works with max_align_t
	assert(sizeof(max_align_t) == sizeof(long double));
        max_align_t* val = (max_align_t*) MALLOC(sizeof(max_align_t));
        *val = id;
        resulting_ids[id] = (int) *val;
        FREE(val);
	*/
    } else {
        int* val = (int*) MALLOC(sizeof(int));
        *val = id;
        resulting_ids[id] = *val;
        FREE(val);
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
    int threads_per_block = 32;
    run_test("basic", blocks, threads_per_block, test);
    run_test("different sizes", blocks, threads_per_block, test_different_size);
    run_test("different types", blocks, threads_per_block, test_different_types);


    return 0;
}
