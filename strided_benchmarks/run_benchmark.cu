#include <functional>

/**
 * Runs benchmark function a number of times with warmup and measures mean and max executions times over all threads and runs
 *
 * @param num_runs How many times the benchmarks should be run for evalutaion
 * @param num_warmup Number of warmup runs
 * @param mean_runtimes Array of size num_runs to store the mean runtime of the threads for each run
 * @param max_runtimes Array of size num_runs to store the max runtime of the threads for each run
 * @param blocks Number of thread blocks to use
 * @param threads_per_block Number of threads per block to use
 * @param run_benchmark Function (runtime, blocks, threads_per_block) -> void that runs benchmark and stores each threads measured runtime into
 *                      runtime[blockIdx.x * blockDum.x + threadIdx.x]
 */
void run_benchmark(int num_runs, int num_warmup, double* mean_runtimes, double* max_runtimes, int blocks, int threads_per_block, std::function<void(clock_t*, int, int)> run_benchmark) {

        int total_threads = blocks * threads_per_block;

        // setup array with runtimes
        clock_t* runtimes_device;
        cudaMalloc((void**)&runtimes_device, total_threads * sizeof(clock_t));

        for (int i = 0; i < num_runs + num_warmup; i++) {

                // run benchmark, wait for exec to finish
                run_benchmark(runtimes_device, blocks, threads_per_block);
                cudaDeviceSynchronize();

		// check for error
		cudaError_t err = cudaGetLastError();
		if (err != cudaSuccess) {
		   std::cout << "CUDA ERROR: " << cudaGetErrorString(err) << std::endl;
		   exit(-1);
     		}

                if (i >= num_warmup) {
                        // retrieve results
                        int run_id = i - num_warmup;
                        clock_t runtimes_host[total_threads];
                        if (cudaMemcpy(runtimes_host, runtimes_device, total_threads * sizeof(clock_t), cudaMemcpyDeviceToHost) != cudaSuccess) {
				std::cout << "cudaMemcpy failed" << std::endl;
				exit(-1);
			}
                	cudaDeviceSynchronize();

                        double sum = 0;
                        double max_time = 0;
                        for (int j = 0; j < total_threads; j++) {
				assert(runtimes_host[j] >= 0);
                                sum += runtimes_host[j];
                                max_time = std::max(max_time, (double) runtimes_host[j]);
                        }
                        mean_runtimes[run_id] = sum / total_threads;
                        max_runtimes[run_id] = max_time;
                }
        }

        // free device memory
        cudaFree(runtimes_device);
}
