#include <iostream>
#include "dynamic_allocator.cu"
#include "../strided_benchmarks/run_benchmark.cu"


__global__ void strided_write_baseline(int num_floats, clock_t* runtime) {
	int id = (blockIdx.x*blockDim.x + threadIdx.x);
	float* ptr = (float*)malloc_baseline(num_floats * sizeof(float));

	clock_t start_time = clock();

	for (int i = 0; i < num_floats; i++) {
		ptr[i] = id + i;
	}

	clock_t end_time = clock();
	runtime[id] = end_time - start_time;
	free_baseline(ptr);
}

__global__ void strided_write_v1(int num_floats, clock_t* runtime) {
    int id = (blockIdx.x*blockDim.x + threadIdx.x);
    float* ptr = (float*)malloc_v1(num_floats * sizeof(float));

    clock_t start_time = clock();

    for (int i = 0; i < num_floats; i++) {
        ptr[i] = id + i;
    }

    clock_t end_time = clock();
    runtime[id] = end_time - start_time;
    free_v1(ptr);
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
	

	// run baseline
	run_benchmark(num_runs, num_warmup, mean_runtimes, max_runtimes, blocks, threads_per_block,
			[num_floats](clock_t* runtimes, int b, int t) -> void {
				strided_write_baseline<<<b, t>>>(num_floats, runtimes);
			}
		     );
	print_arr(mean_runtimes, num_runs);

	// run v1
    run_benchmark(num_runs, num_warmup, mean_runtimes, max_runtimes, blocks, threads_per_block,
            [num_floats](clock_t* runtimes, int b, int t) -> void {
                strided_write_v1<<<b, t>>>(num_floats, runtimes);
            }
             );
    print_arr(mean_runtimes, num_runs);	

	return 0;
}
