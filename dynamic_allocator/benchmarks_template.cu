#include <iostream>
#include "dynamic_allocator.cu"
#include "benchmarks_separate.cu"


// *** Workloads ***

// initialize the malloced space with n incrementing floats
__device__ void init_inc(int n, float* ptr) {
	for (int i = 0; i < n; ++i) {
		ptr[i] = i;
	}
}

// calc sum and store result in first array entry
__device__ void sum_reduce(int n, float* ptr) {
	float res = 0;
	for (int i = 0; i < n; ++i) {
        res += ptr[i];
    }
	ptr[0] = res;
}

// calc product and store result in first array entry
__device__ void prod_reduce(int n, float* ptr) {
    float res = 0;
    for (int i = 0; i < n; ++i) {
        res *= ptr[i];
    }
    ptr[0] = res;
}

// calc max and store result in first array entry
__device__ void max_reduce(int n, float* ptr) {
    float res = 0;
    for (int i = 0; i < n; ++i) {
        if (res < ptr[i]) res = ptr[i];
    }
    ptr[0] = res;
}

// pairwise products
__device__ void pair_prod(int n, float* ptr) {
    for (int i = 0; i < n - 1; ++i) {
		ptr[i] = ptr[i] * ptr[i+1];
    }
}

// sum over all products (O(n^2))
__device__ void sum_all_prod(int n, float* ptr) {
    float res = 0;
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j) {
    		res += ptr[i] * ptr[j];
	    }
	}
	ptr[0] = res;
}
// *** end Workloads ***


// measure 3 individual times for malloc/free and workload
__global__ void sum_reduce_baseline(int num_floats, clock_t* runtime_malloc, clock_t* runtime_work, clock_t* runtime_free) {
    int id = (blockIdx.x*blockDim.x + threadIdx.x);
    
    clock_t start_malloc = clock64();
    float* ptr = (float*)malloc_baseline(num_floats * sizeof(float));
    //printf("ptr_0, block %i: %p\n", blockIdx.x, ptr);
    clock_t end_malloc = clock64();
    runtime_malloc[id] = end_malloc - start_malloc;
    
	init_inc(num_floats, ptr);
    
    clock_t start_work = clock64();
	${WORKLOAD}(num_floats, ptr);
    clock_t end_work = clock64();
    runtime_work[id] = end_work - start_work;
    
    clock_t start_free = clock64();
    free_baseline(ptr);
    clock_t end_free = clock64();
    runtime_free[id] = end_free - start_free;
}

__global__ void sum_reduce_v1(int num_floats, clock_t* runtime_malloc, clock_t* runtime_work, clock_t* runtime_free) {
    int id = (blockIdx.x*blockDim.x + threadIdx.x);
    
    clock_t start_malloc = clock64();
    float* ptr = (float*)malloc_v1(num_floats * sizeof(float));
    //printf("ptr_1, block %i: %p\n", blockIdx.x, ptr);
    clock_t end_malloc = clock64();
    runtime_malloc[id] = end_malloc - start_malloc;
    
	init_inc(num_floats, ptr);
    
    clock_t start_work = clock64();
	${WORKLOAD}(num_floats, ptr);
    clock_t end_work = clock64();
    runtime_work[id] = end_work - start_work;
    
    clock_t start_free = clock64();
    free_v1(ptr);
    clock_t end_free = clock64();
    runtime_free[id] = end_free - start_free;
}

__global__ void sum_reduce_v3(int num_floats, clock_t* runtime_malloc, clock_t* runtime_work, clock_t* runtime_free) {
    int id = (blockIdx.x*blockDim.x + threadIdx.x);
    
    clock_t start_malloc = clock64();
    init_malloc_v3();
    float* ptr = (float*)malloc_v3(num_floats * sizeof(float));
    clock_t end_malloc = clock64();
    runtime_malloc[id] = end_malloc - start_malloc;
    
	init_inc(num_floats, ptr);
    
    clock_t start_work = clock64();
	${WORKLOAD}(num_floats, ptr);
    clock_t end_work = clock64();
    runtime_work[id] = end_work - start_work;
    
    clock_t start_free = clock64();
    free_v3(ptr);
    clean_malloc_v3();
    clock_t end_free = clock64();
    runtime_free[id] = end_free - start_free;
}

void print_arr(double* arr, int len) {
	for (int i = 0; i < len; i++) {
		std::cout << arr[i] << " ";
	}
	std::cout << std::endl;
}

int main(int argc, char **argv) {
	
	// read arguments
	int blocks = atoi(argv[1]);
	int threads_per_block = atoi(argv[2]);
	int num_runs = atoi(argv[3]);
	int num_warmup = atoi(argv[4]);
	int num_floats = atoi(argv[5]);

    // setup of individual measurement arrays
    double mean_runtimes_malloc[num_runs];
    double mean_runtimes_work[num_runs];
    double mean_runtimes_free[num_runs];
    double max_runtimes_malloc[num_runs];
    double max_runtimes_work[num_runs];
    double max_runtimes_free[num_runs];

    cudaDeviceSetLimit(cudaLimitMallocHeapSize, 1000000000); // 1GB
    
    run_benchmark_separate(num_runs, num_warmup, mean_runtimes_malloc, mean_runtimes_work, mean_runtimes_free, max_runtimes_malloc, max_runtimes_work, max_runtimes_free, blocks, threads_per_block, [num_floats](clock_t* runtimes_malloc, clock_t* runtimes_work, clock_t* runtimes_free, int b, int t) -> void {
        ${VERSION}<<<b, t>>>(num_floats, runtimes_malloc, runtimes_work, runtimes_free);
    });
    for(int i = 0; i < num_runs; i++) {
	std::cout << mean_runtimes_malloc[i] << " " << max_runtimes_malloc[i] << " " << mean_runtimes_free[i] << " " << max_runtimes_free[i] << " " << mean_runtimes_work[i] << " " << max_runtimes_work[i] << std::endl;
    }
    return 0;
}
