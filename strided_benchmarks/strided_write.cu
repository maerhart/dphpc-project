#include <functional>
#include <iostream>
#include <assert.h>
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
	if (end_time < start_time) {
		printf("Clock overflow\n");
		assert(false);
	}
}

/**
 * Allocate one array for each warp and write floats into it, leaving unused space between different threads' array segments
 *
 * @param empty_padding How many 4-byte segments to leave empty at end of thread segment
 * @param arr Pointer to allocated array
 * @param num_floats Number of floats to write
 * @param runtime Array to hold the runtime of each thread
 */
__global__ void strided_write_warp_malloc(int empty_padding, int num_floats, clock_t* runtime) {

	// for each block build array that holds shared warp arrays for each warp
	__shared__ float** warp_arrs;
	if (threadIdx.x == 0) {
		warp_arrs = (float**) malloc(sizeof(float*) * ((blockDim.x / 32) + 1));
		if (!warp_arrs) {
			printf("Failed to malloc shared warp array pointers");
		}
	}
	__syncthreads();

	// allocate shared array for entire warp
	int lane_id = threadIdx.x % 32;
	int warp_id = threadIdx.x / 32;
	if (lane_id == 0) {
		float* warp_arr = (float*) malloc(32 * sizeof(float) * (empty_padding + num_floats));
		if (!warp_arr) {
			printf("Failed to malloc %d floats\n", 32 * (empty_padding + num_floats));
		}
		assert(warp_arrs != NULL);
		warp_arrs[warp_id] = warp_arr;
	}
	__syncthreads();
	assert(warp_arrs != NULL);
	assert(warp_arrs[warp_id] != NULL);

	int id = (blockIdx.x*blockDim.x + threadIdx.x);
	float* ptr = warp_arrs[warp_id] + lane_id * (empty_padding + num_floats);

	clock_t start_time = clock();

	for (int i = 0; i < num_floats; i++) {
		ptr[i] = id + i;
	}

	clock_t end_time = clock();
	runtime[id] = end_time - start_time;
	if (end_time < start_time) {
		printf("Clock overflow\n");
		assert(false);
	}

	__syncthreads();
	if (lane_id == 0) {
		free(warp_arrs[warp_id]);
	}
	if (threadIdx.x == 0) {
		free(warp_arrs);
	}
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

	// allocate shared array for entire block
	__shared__ float* block_arr;
	if (threadIdx.x == 0) {
		block_arr = (float*) malloc(blockDim.x * sizeof(float) * (empty_padding + num_floats));
		if (!block_arr) {
			printf("Failed to malloc %d floats\n", blockDim.x * (empty_padding + num_floats));
		}
	}
	__syncthreads();
	assert(block_arr != NULL);

	int id = (blockIdx.x*blockDim.x + threadIdx.x);
	float* ptr = block_arr + threadIdx.x * (empty_padding + num_floats);

	clock_t start_time = clock();

	for (int i = 0; i < num_floats; i++) {
		ptr[i] = id + i;
	}

	clock_t end_time = clock();
	runtime[id] = end_time - start_time;
	if (end_time < start_time) {
		printf("Clock overflow\n");
		assert(false);
	}

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
	if (!ptr) {
		printf("Failed to malloc %d floats\n", (empty_padding + num_floats));
		assert(false);
	}


	clock_t start_time = clock();

	for (int i = 0; i < num_floats; i++) {
		ptr[i] = id + i;
	}

	clock_t end_time = clock();
	runtime[id] = end_time - start_time;
	if (end_time < start_time) {
		printf("Clock overflow\n");
		assert(false);
	}

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

	// run warp malloc write
	run_benchmark(num_runs, num_warmup, mean_runtimes, max_runtimes, blocks, threads_per_block,
			[empty_padding, num_floats](clock_t* runtimes, int b, int t) -> void {
				strided_write_warp_malloc<<<b, t>>>(empty_padding, num_floats, runtimes);
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
