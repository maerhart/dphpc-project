#include <functional>
#include <algorithm>

/**
 * Runs benchmark function a number of times with warmup and measures mean executions times over all threads and runs individually for malloc, work and free
 *
 * @param num_runs How many times the benchmarks should be run for evalutaion
 * @param num_warmup Number of warmup runs
 * @param mean_runtimes_malloc Array of size num_runs to store the mean runtime of the threads for each run of malloc
 * @param mean_runtimes_work Array of size num_runs to store the mean runtime of the threads for each run of workload
 * @param mean_runtimes_free Array of size num_runs to store the mean runtime of the threads for each run of free
 * @param blocks Number of thread blocks to use
 * @param threads_per_block Number of threads per block to use
 * @param run_benchmark Function (runtime, blocks, threads_per_block) -> void that runs benchmark and stores each threads measured runtime into
 *                      runtime[blockIdx.x * blockDum.x + threadIdx.x]
 */
void run_benchmark_separate(int num_runs, int num_warmup, double* mean_runtimes_malloc, double* mean_runtimes_work, double* mean_runtimes_free, double* max_runtimes_malloc, double* max_runtimes_work, double* max_runtimes_free, int blocks, int threads_per_block, std::function<void(clock_t*, clock_t*, clock_t*, int, int)> run_benchmark) {

        int total_threads = blocks * threads_per_block;

        // setup arrays with runtimes
        clock_t* runtimes_device_malloc;
        cudaMalloc((void**)&runtimes_device_malloc, total_threads * sizeof(clock_t));
        clock_t* runtimes_device_work;
        cudaMalloc((void**)&runtimes_device_work, total_threads * sizeof(clock_t));
        clock_t* runtimes_device_free;
        cudaMalloc((void**)&runtimes_device_free, total_threads * sizeof(clock_t));

        for (int i = 0; i < num_runs + num_warmup; i++) {

                // run benchmark, wait for exec to finish
                run_benchmark(runtimes_device_malloc, runtimes_device_work, runtimes_device_free, blocks, threads_per_block);

                cudaDeviceSynchronize();

                
                if (i >= num_warmup) {
                        // retrieve results
                        int run_id = i - num_warmup;
                        clock_t* runtimes_host_malloc = (clock_t*)malloc(total_threads * sizeof(clock_t));
                        cudaMemcpy(runtimes_host_malloc, runtimes_device_malloc, total_threads * sizeof(clock_t), cudaMemcpyDeviceToHost);
                        clock_t* runtimes_host_work = (clock_t*)malloc(total_threads * sizeof(clock_t));
                        cudaMemcpy(runtimes_host_work, runtimes_device_work, total_threads * sizeof(clock_t), cudaMemcpyDeviceToHost);
                        clock_t* runtimes_host_free = (clock_t*)malloc(total_threads * sizeof(clock_t));
                        cudaMemcpy(runtimes_host_free, runtimes_device_free, total_threads * sizeof(clock_t), cudaMemcpyDeviceToHost);
                        
                        cudaError_t err = cudaGetLastError();
                        if (err != cudaSuccess)
                        {
                            const char * errorMessage = cudaGetErrorString(err);
                            printf("CUDA error: %s \n", errorMessage);
                        }
                    
                        double sum_malloc = 0;
                        double sum_work = 0;
                        double sum_free = 0;
                        double max_malloc = 0;
                        double max_work = 0;
                        double max_free = 0;
                    
                        // work over all
                        for (int j = 0; j < total_threads; j++) {
                                // mean
                                sum_malloc += runtimes_host_malloc[j];
                                sum_work += runtimes_host_work[j];
                                sum_free += runtimes_host_free[j];
                            
                                // max
                                max_malloc = std::max(max_malloc, (double)runtimes_host_malloc[j]);
                                max_work = std::max(max_work, (double)runtimes_host_work[j]);
                                max_free = std::max(max_free, (double)runtimes_host_free[j]);
                        }
                        mean_runtimes_malloc[run_id] = sum_malloc / total_threads;
                        mean_runtimes_work[run_id] = sum_work / total_threads;
                        mean_runtimes_free[run_id] = sum_free / total_threads;
                    
                        max_runtimes_malloc[run_id] = max_malloc;
                        max_runtimes_work[run_id] = max_work;
                        max_runtimes_free[run_id] = max_free;
                }
        }

        // free device memory
        cudaFree(runtimes_device_malloc);
        cudaFree(runtimes_device_work);
        cudaFree(runtimes_device_free);
}