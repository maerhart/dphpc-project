#include <iostream>
#include "dynamic_allocator.cu"
#include "../strided_benchmarks/run_benchmark.cu"


__global__ void strided_write_baseline(int num_floats, clock_t* runtime) {
	int id = (blockIdx.x*blockDim.x + threadIdx.x);
	float* ptr = (float*)malloc_baseline(num_floats * sizeof(float));
	//printf("%ld\n", ptr);
	__syncthreads();
	clock_t start_time = clock();

	for (int i = 0; i < num_floats; i++) {
		ptr[i] = id + i;
	}

	clock_t end_time = clock();
	__syncthreads();
	//printf("%ld\n", end_time - start_time);
	//runtime[id] = (double)(end_time - start_time) / (double)CLOCKS_PER_SEC;
	runtime[id] = end_time - start_time;
	free_baseline(ptr);
}

__global__ void strided_write_v1(int num_floats, clock_t* runtime) {
    int id = (blockIdx.x*blockDim.x + threadIdx.x);
	float* ptr = (float*)malloc_v1(num_floats * sizeof(float));
    // check if pointers overlap with runtime
	//printf("Runtime ptr: %p, Malloc ptr: %p\n", runtime, ptr);

	//printf("%p\n", (void*)ptr);
	__syncthreads();
	clock_t start_time = clock();
	
    for (int i = 0; i < num_floats; i++) {
        ptr[i] = id + i;
    }

    clock_t end_time = clock();
	__syncthreads();
    //printf("%ld\n", end_time - start_time);
	//runtime[id] = (double)(end_time - start_time) / (double)CLOCKS_PER_SEC;
	runtime[id] = end_time - start_time;
	//printf("%f\n", runtime[id]);
    free_v1(ptr);
}

__global__ void strided_write_v2(int num_floats, clock_t* runtime) {
    int id = (blockIdx.x*blockDim.x + threadIdx.x);
    float* ptr = (float*)malloc_v2(num_floats * sizeof(float));
    // check if pointers overlap with runtime
    //printf("Runtime ptr: %p, Malloc ptr: %p\n", runtime, ptr);

    //printf("%p\n", (void*)ptr);
    __syncthreads();
    clock_t start_time = clock();

    for (int i = 0; i < num_floats; i++) {
        ptr[i] = id + i;
    }

    clock_t end_time = clock();
    __syncthreads();
    //printf("%ld\n", end_time - start_time);
    //runtime[id] = (double)(end_time - start_time) / (double)CLOCKS_PER_SEC;
    runtime[id] = end_time - start_time;
    //printf("%f\n", runtime[id]);
    free_v2(ptr);
}

void print_arr(double* arr, int len) {
	for (int i = 0; i < len; i++) {
		std::cout << arr[i] << " ";
	}
	std::cout << std::endl;
}

int main(int argc, char **argv) {
	
	// read args
	int blocks = atoi(argv[1]);
	int threads_per_block = atoi(argv[2]);
	int num_runs = atoi(argv[3]);
	int num_warmup = atoi(argv[4]);
	int num_floats = atoi(argv[5]);

	// setup measurement arrays
	double mean_runtimes[num_runs];
	double max_runtimes[num_runs];
	

	/*
	// run benchmarks
	double mean = 0, max = 0;
	int total_threads = blocks * threads_per_block;
	
	for (int i = 0; i < num_runs; ++i) {
		double* runtime_per_thread;
		//double* d_runtime_per_thread;
		int size_runtimes = total_threads * sizeof(double);
		//runtime_per_thread = (double*)malloc(size_runtimes);
		cudaMallocManaged(&runtime_per_thread, size_runtimes);

		cudaDeviceSetLimit(cudaLimitMallocHeapSize, 1000000000); // 1GB
		strided_write_baseline<<<blocks, threads_per_block>>>(num_floats, runtime_per_thread);
		cudaDeviceSynchronize();

		//cudaMemcpy(runtime_per_thread, d_runtime_per_thread, size_runtimes, cudaMemcpyDeviceToHost);
		//cudaDeviceSynchronize();
		double mean_run = 0;
		for (int j = 0; j < total_threads; ++j) {
			mean_run += runtime_per_thread[j];
			//printf("%f\n", runtime_per_thread[j]);
		}
		mean_run /= total_threads;

		mean += mean_run;
	}
	mean /= num_runs;
	std::cout << mean << std::endl;
	*/	


	cudaDeviceSetLimit(cudaLimitMallocHeapSize, 1000000000); // 1GB
	
	// run baseline
	run_benchmark(num_runs, num_warmup, mean_runtimes, max_runtimes, blocks, threads_per_block,
			[num_floats](clock_t* runtimes, int b, int t) -> void {
				strided_write_baseline<<<b, t>>>(num_floats, runtimes);
			}
		     );
	//print_arr(mean_runtimes, num_runs);

	double mean_baseline = 0;
	for (int i = 0; i < num_runs; ++i) {
		mean_baseline += mean_runtimes[i];
	}
	mean_baseline /= num_runs;
	std::cout << mean_baseline << std::endl;
	


	// run v1
    run_benchmark(num_runs, num_warmup, mean_runtimes, max_runtimes, blocks, threads_per_block,
            [num_floats](clock_t* runtimes, int b, int t) -> void {
                strided_write_v1<<<b, t>>>(num_floats, runtimes);
            }
             );
	cudaDeviceSynchronize();
    //print_arr(mean_runtimes, num_runs);	

	double mean_v1 = 0;
    for (int i = 0; i < num_runs; ++i) {
        mean_v1 += mean_runtimes[i];
    }
    mean_v1 /= num_runs;
    std::cout << mean_v1 << std::endl;


	// run v2
    run_benchmark(num_runs, num_warmup, mean_runtimes, max_runtimes, blocks, threads_per_block,
            [num_floats](clock_t* runtimes, int b, int t) -> void {
                strided_write_v2<<<b, t>>>(num_floats, runtimes);
            }
             );
    cudaDeviceSynchronize();
    //print_arr(mean_runtimes, num_runs);

    double mean_v2 = 0;
    for (int i = 0; i < num_runs; ++i) {
        mean_v2 += mean_runtimes[i];
    }
    mean_v2 /= num_runs;
    std::cout << mean_v2 << std::endl;

	
	return 0;
}
