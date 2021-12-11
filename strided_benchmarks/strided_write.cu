#include <functional>
#include <iostream>
#include "run_benchmark.cu"




/**
 * Write floats into a given array, leaving unused space between different threads' array segments
 *
 * @param empty_padding How many 4-byte segments to leave empty at end of thread segment
 * @param arr Pointer to allocated array
 * @param num_floats Number of floats to write
 * @param runtime Array to hold the runtime of each thread
 */
__global__ void strided_write(int empty_padding, float* arr, int num_floats, clock_t* runtime) {
	int id = (blockIdx.x*blockDim.x + threadIdx.x);
	float* ptr = arr + id * (empty_padding + num_floats);

	clock_t start_time = clock();

	for (int i = 0; i < num_floats; i++) {
		ptr[i] = id + i;
	}

	clock_t end_time = clock();
	runtime[id] = end_time - start_time;
}

/**
 * Allocate one array for each block and write floats into it, leaving unused space between different threads' array segments
 *
 * @param empty_padding How many 4-byte segments to leave empty at end of thread segment
 * @param arr Pointer to allocated array
 * @param num_floats Number of floats to write
 * @param runtime Array to hold the runtime of each thread
 */
__global__ void strided_write_block_malloc(int empty_padding, int num_floats, clock_t* runtime) {

	__shared__ float* block_arr;

	if (threadIdx.x == 0) {
		block_arr = (float*) malloc(blockDim.x * sizeof(float) * (empty_padding + num_floats));
	}
	__syncthreads();

	int id = (blockIdx.x*blockDim.x + threadIdx.x);
	float* ptr = block_arr + id * (empty_padding + num_floats);

	clock_t start_time = clock();

	for (int i = 0; i < num_floats; i++) {
		ptr[i] = id + i;
	}

	clock_t end_time = clock();
	runtime[id] = end_time - start_time;

	__syncthreads();
	if (threadIdx.x == 0) {
		free(block_arr);
	}
}

/**
 * Malloc an array and Write floats into it
 *
 * @param empty_padding How many 4-byte segments to leave empty at end of thread's malloc segement
 * @param num_floats Number of floats to write
 * @param runtime Array to hold the runtime of each thread
 */
__global__ void strided_write_all_malloc(int empty_padding, int num_floats, clock_t* runtime) {
	int id = (blockIdx.x*blockDim.x + threadIdx.x);
	float* ptr = (float*) malloc(sizeof(float) * (num_floats + empty_padding));


	clock_t start_time = clock();

	for (int i = 0; i < num_floats; i++) {
		ptr[i] = id + i;
	}

	clock_t end_time = clock();
	runtime[id] = end_time - start_time;

	free(ptr);
}

void run_strided_write(clock_t* runtime, int blocks, int threads_per_block, int empty_padding, int num_floats) {
	int total_threads = blocks * threads_per_block;

	// setup array to write to
	float* arr;
	cudaMalloc((void**)&arr, total_threads * sizeof(float) * (num_floats + empty_padding));

	strided_write<<<blocks, threads_per_block>>>(empty_padding, arr, num_floats, runtime);

	cudaFree(arr);

}

void print_arr(double* arr, int len) {
	for (int i = 0; i < len; i++) {
		std::cout << arr[i] << " ";
	}
	std::cout << std::endl;
}

/**
 * Args: blocks, threads_per_block, num_runs, num_warmup, empty_padding, num_floats
 */
int main(int argc, char **argv) {
	// read args
	int blocks = atoi(argv[1]);
	int threads_per_block = atoi(argv[2]);
	int num_runs = atoi(argv[3]);
	int num_warmup = atoi(argv[4]);
	int empty_padding = atoi(argv[5]);
	int num_floats = atoi(argv[6]);

	// setup measurement arrays
	double mean_runtimes[num_runs];
	double max_runtimes[num_runs];

	// run strided write
	run_benchmark(num_runs, num_warmup, mean_runtimes, max_runtimes, blocks, threads_per_block,
			[empty_padding, num_floats](clock_t* runtimes, int b, int t) -> void {
				run_strided_write(runtimes, b, t, empty_padding, num_floats);
			}
		     );
	print_arr(mean_runtimes, num_runs);
	//print_arr(max_runtimes, num_runs);

	// run block malloc write
	run_benchmark(num_runs, num_warmup, mean_runtimes, max_runtimes, blocks, threads_per_block,
			[empty_padding, num_floats](clock_t* runtimes, int b, int t) -> void {
				strided_write_block_malloc<<<b, t>>>(empty_padding, num_floats, runtimes);
			}
		     );
	print_arr(mean_runtimes, num_runs);
	//print_arr(max_runtimes, num_runs);

	// run all malloc write
	run_benchmark(num_runs, num_warmup, mean_runtimes, max_runtimes, blocks, threads_per_block,
			[empty_padding, num_floats](clock_t* runtimes, int b, int t) -> void {
				strided_write_all_malloc<<<b, t>>>(empty_padding, num_floats, runtimes);
			}
		     );
	print_arr(mean_runtimes, num_runs);
	//print_arr(max_runtimes, num_runs);
}