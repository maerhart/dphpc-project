/*
    Benchmark for the point-to-point protocol of memory copy between threads of the same GPU.
    Copies are going through GPU global memory.
    Each thread have pointer to its own memory located in CUDA global memory space. 
    One of threads issues send operation with following wait. The other thread issues recv operation with following wait.
    Benchmark measures total time of data transfer and tries to determine throughput and latency from it.
*/
#include <cuda.h>

#include <new>
#include <cstdio>
#include <chrono>
#include <cassert>

#define CHECK(expr) \
    do { if ((expr) != 0) { printf("ERROR %s:%d %s\n", __FILE__, __LINE__, #expr); abort(); } } while (0)

__global__ void copyKernel(void* dst, void* src, size_t size) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    char* d = (char*) dst;
    char* s = (char*) src;
    for (int i = idx; i < size; i += blockDim.x * gridDim.x) {
        d[i] = s[i];
    }
}

static const int measurements = 23; 
const size_t maxBufferSize = 1 << measurements; 
const int numBenchmarks = 5;
const char* benchmarkName[5] = {
    "sender_issues_transfer",
    "host_issues_tranfer_memcpy",
    "host_issues_tranfer_copy_kernel",
    "device_issues_tranfer_memcpy",
    "device_issues_tranfer_copy_kernel"
};
const double interval = 0.1;
const int clockWarmup = 10;
const double minMeasurementTime = 0.1;
const int stages = 2;


cudaStream_t hostCopyStream;
cudaStream_t benchmarkStream;

__forceinline__ __host__ __device__ void memfence() {
    #if __CUDA_ARCH__
        __threadfence_system();
    #else
        __sync_synchronize();
    #endif
}

struct SharedState {
    SharedState() {
        CHECK(cudaMalloc(&buffer1, maxBufferSize));
        CHECK(cudaMalloc(&buffer2, maxBufferSize));

        CHECK(cudaMemset(buffer1, 0, maxBufferSize));
        CHECK(cudaMemset(buffer2, 0, maxBufferSize));

        for (int b = 0; b < numBenchmarks; b++) {
            for (int m = 0; m < measurements; m++) {
                repetitions[b][m] = 1;
            }
        }
    }

    // clock rate related 
    bool clockMeasurementGPUReady = false;
    bool clockMeasurementCPUReady = false;
    long long clocksPerSecond = -1;

    __device__ double dt(long long t1, long long t2) {
        return (t2 - t1) * 1.0 / clocksPerSecond;
    }

    // memory buffers
    char* buffer1 = nullptr;
    char* buffer2 = nullptr;

    volatile bool thread1ready = false;
    volatile bool thread2ready = false;
    volatile bool cpuReady = false;
    
    // implementation of send/recv for two threads
    void* volatile sendPtr = nullptr;
    void* volatile recvPtr = nullptr;
    volatile size_t transferSize = 0;

    volatile bool sendCompleted = false;
    volatile bool recvCompleted = false;

    double times[numBenchmarks][measurements];
    int repetitions[numBenchmarks][measurements];
    
    __device__ int size() {
        return blockDim.x * gridDim.x;
    }

    __device__ int rank() {
        return threadIdx.x + blockDim.x * blockIdx.x;
    }

    __device__ void send(void* buf, size_t size) {
        while (sendCompleted || recvCompleted) {}
        if (sendPtr) {
            printf("ERROR send\n");
        }
        sendPtr = buf;
        transferSize = size;
    }

    __device__ void recv(void* buf, size_t size) {
        while (recvCompleted || sendCompleted) {}
        if (recvPtr) {
            printf("ERROR recv\n");
        }
        recvPtr = buf;
        transferSize = size;
    }

    __host__ __device__ void waitSendFinish() {
        while (!sendCompleted) {}
        sendCompleted = false;
    }

    __host__ __device__ void waitRecvFinish() {
        while (!recvCompleted) {}
        recvCompleted = false;
    }

    __host__ __device__ void waitTransferArguments() {
        while (!(sendPtr && recvPtr)) {}
    }

    __host__ __device__ void finishTransfer() {
        sendPtr = nullptr;
        recvPtr = nullptr;
        sendCompleted = true;
        recvCompleted = true;
    }

    __device__ void deviceSequentialTransfer() {
        waitTransferArguments();
        memcpy(recvPtr, sendPtr, transferSize);
        finishTransfer();
    }

    __host__ __device__ void copyKernelTransfer() {
        waitTransferArguments();
#if __CUDA_ARCH__
        copyKernel<<<256, 256>>>(recvPtr, sendPtr, transferSize);
        cudaDeviceSynchronize();
#else
        copyKernel<<<256, 256, 0, hostCopyStream>>>(recvPtr, sendPtr, transferSize);
        CHECK(cudaStreamSynchronize(hostCopyStream));
#endif
        finishTransfer();
    }

    __host__ __device__ void memcpyTransfer() {
        waitTransferArguments();
#if __CUDA_ARCH__
        cudaMemcpyAsync(recvPtr, sendPtr, transferSize, cudaMemcpyDefault);
        cudaDeviceSynchronize();
#else
        cudaMemcpyAsync(recvPtr, sendPtr, transferSize, cudaMemcpyDefault, hostCopyStream);
        CHECK(cudaStreamSynchronize(hostCopyStream));
#endif
        finishTransfer();
    }

    // *** barrier between host and single device thread ***

    volatile bool hostBarrierReady = 0;
    volatile bool deviceBarrierReady = 0;

    __device__ void syncWithHost() {
        while (deviceBarrierReady) {} // wait previous entry
        deviceBarrierReady = true;
        while (!hostBarrierReady) {}
        hostBarrierReady = false;
    }

    void syncWithDeviceThread() {
        while (hostBarrierReady) {} // wait previous entry
        hostBarrierReady = true;
        while (!deviceBarrierReady) {}
        deviceBarrierReady = false;
    }

    // *** barrier between first threads of all blocks ***

    unsigned bi = 0;
    unsigned bo = 0;

    __device__ void syncBlocksMasters() {
        int numBlocks = gridDim.x;

        volatile unsigned* vbi = &bi;
        volatile unsigned* vbo = &bo;

        if (threadIdx.x == 0) {
            // wait other threads to exit from previous barrier invocation
            while (*vbo != 0) {}

            unsigned oldIn = atomicAdd_system(&bi, 1);

            // if we are last thread, reset out counter
            // and allow threads to pass barrier entry 
            if (oldIn == numBlocks - 1) {
                *vbo = numBlocks + 1;
                //memfence();
                *vbi += 1; // increase second time to numBlocks + 1
                //memfence();
            }
            
            // barrier entry
            while (*vbi != numBlocks + 1) {} 

            // if we are here, then all threads started exitting from barrier
            unsigned oldOut = atomicSub_system(&bo, 1);
            if (oldOut == 2) {
                *vbi = 0;
                //memfence();
                *vbo -= 1; // decrease second time to 0
                //memfence();
            }
        }
    }

    // barrier between all threads on device
    __device__ void deviceBarrier() {
        // Since synchtreads is not enough for correct barrier: 
        // one should be in the beginning and one at the end/
        __syncthreads();
        syncBlocksMasters();
        __syncthreads();
    }

    // barrier between all threads on device and host thread
    __host__ __device__ void hostDeviceBarrier() {
#ifdef __CUDA_ARCH__
        __syncthreads();
        if (rank() == 0) syncWithHost();
        syncBlocksMasters();
        if (rank() == 0) syncWithHost();
        __syncthreads();
#else
        // twice, it is not a bug!
        syncWithDeviceThread();
        syncWithDeviceThread();
#endif
    }

};

__global__ void clockMeasurementKernel(SharedState* ss) {
    volatile bool& cpuReady = ss->clockMeasurementCPUReady;
    volatile bool& gpuReady = ss->clockMeasurementGPUReady;

    for (int i = 0; i < clockWarmup; i++) {
        memfence();
        while (!cpuReady) {}
        gpuReady = true;
        long long t1 = clock64();
        memfence();
        while (cpuReady) {}
        long long t2 = clock64();
        gpuReady = false;

        printf("%lld clocks per %lg seconds\n", t2 - t1, interval);
        ss->clocksPerSecond = (t2 - t1) / interval; // use value from last iteration
    }
}

__global__ void checkClockRateCalibration(SharedState* ss) {
    // check clock rate calibration
    long long t1 = clock64();
    while (ss->dt(t1, clock64()) < interval) {}
    long long t2 = clock64();
    double dt = ss->dt(t1, t2);
    printf("dt on GPU = %lg s\n", dt);
}

// https://en.wikipedia.org/wiki/Ordinary_least_squares#Simple_linear_regression_model
void leastSquares(int n, const double* x, const double* y, double& a, double& b) {
    //printf("+++ simple linear regression +++\n");
    //for (int i = 0; i < n; i++) {
    //    printf("x[%d] = %lg y[%d] = %lg\n", i, x[i], i, y[i]);
    //}
    double meanX = 0;
    double meanY = 0;
    for (int i = 0; i < n; i++) {
        meanX += x[i];
        meanY += y[i];
    }
    meanX /= n;
    meanY /= n;
    // hack: force least squares to consider impact of the first point more 
    // it is required due to exponential growth of data points x
    meanX = x[0];
    meanY = y[0];

    double covXY = 0;
    double varX = 0;
    for (int i = 0; i < n; i++) {
        double dx = x[i] - meanX;
        double dy = y[i] - meanY;
        covXY += dx * dy;
        varX += dx * dx;
    }
    a = covXY / varX;
    b = meanY - a * meanX;
    //printf("meanX = %lg\n", meanX);
    //printf("meanY = %lg\n", meanY);
    //printf("a = %lg\n", a);
    //printf("b = %lg\n", b);
    //printf("=== simple linear regression ===\n");
}

__global__ void benchmarkKernel(SharedState* ss) {
    int rank = ss->rank();
    double times[measurements];
    int benchmark = 0;

    // benchmark 1: sender issues transfer
    for (int stage = 0; stage < stages; stage++) {
        //printf("rank = %d stage = %d\n", rank, stage);
        //printf("rank = %d stage = %d repetitions[0] = %d\n", rank, stage, ss->repetitions[benchmark][0]);
        //volatile int* vreps = &(ss->repetitions[benchmark][0]);
        //printf("rank = %d stage = %d volatile repetitions[0] = %d\n", rank, stage, *vreps);

        //long long t1 = clock64();
        //while (clock64() - t1 < 2000000000) {}
        //printf("rank = %d stage = %d repetitions[0] = %d\n", rank, stage, ss->repetitions[benchmark][0]);
        //printf("rank = %d stage = %d volatile repetitions[0] = %d\n", rank, stage, *vreps);

        for (int m = 0; m < measurements; m++) {
            size_t bufferSize = 1 << m;
            int repeats = ss->repetitions[benchmark][m];
            int warmup = repeats / 10 + 1;
            long long t1 = 0;
            //printf("rank = %d benchmark %d repetitions %d bufferSize %d\n", rank, benchmark, int(ss->repetitions[benchmark][m]), int(bufferSize));
            for (int r = 0; r < warmup + repeats; r++) { 
                if (r == warmup) {
                    t1 = clock64(); // start measurement
                }
                if (rank == 0) {
                    ss->send(ss->buffer1, bufferSize);
                    ss->deviceSequentialTransfer();
                    ss->waitSendFinish();

                    ss->recv(ss->buffer1, bufferSize);
                    ss->waitRecvFinish();
                } else if (rank == 1) {
                    ss->recv(ss->buffer2, bufferSize);
                    ss->waitRecvFinish();

                    ss->send(ss->buffer2, bufferSize);
                    ss->deviceSequentialTransfer();
                    ss->waitSendFinish();
                }
            }
            long long t2 = clock64();
            times[m] = ss->dt(t1, t2) / repeats;
            if (rank == 0) {
                //printf("rank = %d benchmark %d repetitions %d bufferSize %d time %f bw %f\n", rank, benchmark, int(ss->repetitions[benchmark][m]), int(bufferSize), times[m], bufferSize / times[m]);
                printf("benchmark %d repetitions %d bufferSize %d time %f bw %f\n", benchmark, int(ss->repetitions[benchmark][m]), int(bufferSize), times[m], bufferSize / times[m]);
            }
        }

        ss->hostDeviceBarrier();

        // copy measurements to the host
        if (rank == 0) {
            memcpy(ss->times[benchmark], times, sizeof(double) * measurements);
        }

        ss->hostDeviceBarrier();
        // wait host to update number of repetitions
        ss->hostDeviceBarrier();
    }

    benchmark += 1;

    // benchmark 2: host issues transfer with memcpy

    for (int stage = 0; stage < stages; stage++) {

        for (int m = 0; m < measurements; m++) {
            size_t bufferSize = 1 << m;
            int repeats = ss->repetitions[benchmark][m];
            int warmup = repeats / 10 + 1;
            long long t1 = 0;
            for (int r = 0; r < warmup + repeats; r++) { 
                if (r == warmup) {
                    t1 = clock64(); // start measurement
                }
                if (rank == 0) {
                    ss->send(ss->buffer1, bufferSize);
                    ss->waitSendFinish();

                    ss->recv(ss->buffer1, bufferSize);
                    ss->waitRecvFinish();
                } else if (rank == 1) {
                    ss->recv(ss->buffer2, bufferSize);
                    ss->waitRecvFinish();

                    ss->send(ss->buffer2, bufferSize);
                    ss->waitSendFinish();
                }
            }
            long long t2 = clock64();
            times[m] = ss->dt(t1, t2) / repeats;
            if (rank == 0) {
                printf("benchmark %d repetitions %d bufferSize %d time %f bw %f\n", benchmark, int(ss->repetitions[benchmark][m]), int(bufferSize), times[m], bufferSize / times[m]);
            }
        }

        ss->hostDeviceBarrier();

        // copy measurements to the host
        if (rank == 0) {
            memcpy(ss->times[benchmark], times, sizeof(double) * measurements);
        }

        ss->hostDeviceBarrier();
        // wait host to update number of repetitions
        ss->hostDeviceBarrier();
    }


    //printf("Device wants to start benchmark 2\n");

    benchmark += 1;

    // benchmark 3: host issues transfer copy kernel

    for (int stage = 0; stage < stages; stage++) {

        for (int m = 0; m < measurements; m++) {
            size_t bufferSize = 1 << m;
            int repeats = ss->repetitions[benchmark][m];
            int warmup = repeats / 10 + 1;
            long long t1 = 0;
            for (int r = 0; r < warmup + repeats; r++) { 
                if (r == warmup) {
                    t1 = clock64(); // start measurement
                }
                if (rank == 0) {
                    ss->send(ss->buffer1, bufferSize);
                    ss->waitSendFinish();

                    ss->recv(ss->buffer1, bufferSize);
                    ss->waitRecvFinish();
                } else if (rank == 1) {
                    ss->recv(ss->buffer2, bufferSize);
                    ss->waitRecvFinish();

                    ss->send(ss->buffer2, bufferSize);
                    ss->waitSendFinish();
                }
            }
            long long t2 = clock64();
            times[m] = ss->dt(t1, t2) / repeats;
            if (rank == 0) {
                printf("benchmark %d repetitions %d bufferSize %d time %f bw %f\n", benchmark, int(ss->repetitions[benchmark][m]), int(bufferSize), times[m], bufferSize / times[m]);
            }
        }
        ss->hostDeviceBarrier();

        // copy measurements to the host
        if (rank == 0) {
            memcpy(ss->times[benchmark], times, sizeof(double) * measurements);
        }

        ss->hostDeviceBarrier();
        // wait host to update number of repetitions
        ss->hostDeviceBarrier();
    }

    benchmark += 1;

    // benchmark 4: device issues transfer memcpy
    for (int stage = 0; stage < stages; stage++) {
        for (int m = 0; m < measurements; m++) {
            size_t bufferSize = 1 << m;
            int repeats = ss->repetitions[benchmark][m];
            int warmup = repeats / 10 + 1;
            long long t1 = 0;
            for (int r = 0; r < warmup + repeats; r++) { 
                if (r == warmup) {
                    t1 = clock64(); // start measurement
                }
                if (rank == 0) {
                    ss->send(ss->buffer1, bufferSize);
                    ss->memcpyTransfer();
                    ss->waitSendFinish();

                    ss->recv(ss->buffer1, bufferSize);
                    ss->waitRecvFinish();
                } else if (rank == 1) {
                    ss->recv(ss->buffer2, bufferSize);
                    ss->waitRecvFinish();

                    ss->send(ss->buffer2, bufferSize);
                    ss->memcpyTransfer();
                    ss->waitSendFinish();
                }
            }
            long long t2 = clock64();
            times[m] = ss->dt(t1, t2) / repeats;
            if (rank == 0) {
                printf("benchmark %d repetitions %d bufferSize %d time %f bw %f\n", benchmark, int(ss->repetitions[benchmark][m]), int(bufferSize), times[m], bufferSize / times[m]);
            }
        }
        ss->hostDeviceBarrier();

        // copy measurements to the host
        if (rank == 0) {
            memcpy(ss->times[benchmark], times, sizeof(double) * measurements);
        }

        ss->hostDeviceBarrier();
        // wait host to update number of repetitions
        ss->hostDeviceBarrier();
    }

    benchmark += 1;

    // benchmark 5: device issues transfer copy kernel
    for (int stage = 0; stage < stages; stage++) {

        for (int m = 0; m < measurements; m++) {
            size_t bufferSize = 1 << m;
            int repeats = ss->repetitions[benchmark][m];
            int warmup = repeats / 10 + 1;
            long long t1 = 0;
            for (int r = 0; r < warmup + repeats; r++) { 
                if (r == warmup) {
                    t1 = clock64(); // start measurement
                }
                if (rank == 0) {
                    ss->send(ss->buffer1, bufferSize);
                    ss->copyKernelTransfer();
                    ss->waitSendFinish();

                    ss->recv(ss->buffer1, bufferSize);
                    ss->waitRecvFinish();
                } else if (rank == 1) {
                    ss->recv(ss->buffer2, bufferSize);
                    ss->waitRecvFinish();

                    ss->send(ss->buffer2, bufferSize);
                    ss->copyKernelTransfer();
                    ss->waitSendFinish();
                }
            }
            long long t2 = clock64();
            times[m] = ss->dt(t1, t2) / repeats;
            if (rank == 0) {
                printf("benchmark %d repetitions %d bufferSize %d time %f bw %f\n", benchmark, int(ss->repetitions[benchmark][m]), int(bufferSize), times[m], bufferSize / times[m]);
            }
        }
        ss->hostDeviceBarrier();

        // copy measurements to the host
        if (rank == 0) {
            memcpy(ss->times[benchmark], times, sizeof(double) * measurements);
        }

        ss->hostDeviceBarrier();
        // wait host to update number of repetitions
        ss->hostDeviceBarrier();
    }

    benchmark += 1;
}

using hrclock = std::chrono::high_resolution_clock;

template <typename T>
auto nanoseconds(T x) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(x);
}

int main() {
    SharedState* ss = nullptr;
    CHECK(cudaMallocManaged(&ss, sizeof(SharedState)));
    ss = new (ss) SharedState;

    CHECK(cudaDeviceSynchronize());
    // this is highly unreliable, use measurement approach instead
    //CHECK(cudaDeviceGetAttribute(&ss->peakClkKHz, cudaDevAttrClockRate, /*device = */0)); 

    {
        printf("+++ clock rate calibration +++\n");
        volatile bool& cpuReady = ss->clockMeasurementCPUReady;
        volatile bool& gpuReady = ss->clockMeasurementGPUReady;
        cpuReady = false;
        gpuReady = false;

        // measure gpu clock
        clockMeasurementKernel<<<1,1>>>(ss);
        CHECK(cudaPeekAtLastError());

        for (int i = 0; i < clockWarmup; i++) {
            cpuReady = true;
            memfence();
            while (!gpuReady) {} 
            memfence();
            auto t1 = hrclock::now();
            while (nanoseconds(hrclock::now() - t1).count() / 1e9 < interval) {}
            memfence();
            cpuReady = false;
            memfence();
            while (gpuReady) {}
        }

        CHECK(cudaDeviceSynchronize());
        printf("=== clock rate calibration ===\n");
    }

    { 
        printf("+++ check clock rate calibration +++\n");
        auto t1 = hrclock::now();
        checkClockRateCalibration<<<1, 1>>>(ss);
        CHECK(cudaPeekAtLastError());
        CHECK(cudaDeviceSynchronize());
        auto t2 = hrclock::now();
        double time = nanoseconds(t2 - t1).count() / 1e9;
        printf("dt on host = %lg s (have to be slightly larger than on GPU)\n", time);

        printf("=== check clock rate calibration ===\n");
    }

    CHECK(cudaStreamCreate(&hostCopyStream));
    CHECK(cudaStreamCreate(&benchmarkStream));

    {
        printf("+++ benchmark +++\n");
        benchmarkKernel<<<2, 1, 0, benchmarkStream>>>(ss);
        CHECK(cudaPeekAtLastError());

        int benchmark = 0;

        for (int stage = 0; stage < stages; stage++) {
            // host doesn't participate
            ss->hostDeviceBarrier();
            ss->hostDeviceBarrier();

            // adjust number of repetitions
            for (int m = 0; m < measurements; m++) {
                double time = ss->times[benchmark][m];
                //time = 0.011;
                ss->repetitions[benchmark][m] = minMeasurementTime / time + 1;
                //printf("masurement %d time %lg repetitions %d\n", m, time, ss->repetitions[benchmark][m]);
            }
            memfence();

            ss->hostDeviceBarrier();
        }

        benchmark += 1;

        for (int stage = 0; stage < stages; stage++) {
            for (int m = 0; m < measurements; m++) {
                int repeats = ss->repetitions[benchmark][m];
                int warmup = repeats / 10 + 1;
                for (int r = 0; r < warmup + repeats; r++) { 
                    ss->memcpyTransfer();
                    ss->memcpyTransfer();
                }
            }

            ss->hostDeviceBarrier();
            ss->hostDeviceBarrier();

            // adjust number of repetitions
            for (int m = 0; m < measurements; m++) {
                double time = ss->times[benchmark][m];
                //time = 0.011;
                ss->repetitions[benchmark][m] = minMeasurementTime / time + 1;
                //printf("masurement %d time %lg repetitions %d\n", m, time, ss->repetitions[benchmark][m]);
            }

            ss->hostDeviceBarrier();
        }

        //printf("Host wants to start benchmark 2\n");
        benchmark += 1;

        for (int stage = 0; stage < stages; stage++) {
            for (int m = 0; m < measurements; m++) {
                int repeats = ss->repetitions[benchmark][m];
                int warmup = repeats / 10 + 1;
                for (int r = 0; r < warmup + repeats; r++) { 
                    ss->copyKernelTransfer();
                    ss->copyKernelTransfer();
                }
            }

            ss->hostDeviceBarrier();
            ss->hostDeviceBarrier();

            // adjust number of repetitions
            for (int m = 0; m < measurements; m++) {
                double time = ss->times[benchmark][m];
                //time = 0.011;
                ss->repetitions[benchmark][m] = minMeasurementTime / time + 1;
                //printf("masurement %d time %lg repetitions %d\n", m, time, ss->repetitions[benchmark][m]);
            }

            ss->hostDeviceBarrier();
        }

        benchmark += 1;

        for (int stage = 0; stage < stages; stage++) {
            // host doesn't participate
            ss->hostDeviceBarrier();
            ss->hostDeviceBarrier();

            // adjust number of repetitions
            for (int m = 0; m < measurements; m++) {
                double time = ss->times[benchmark][m];
                //time = 0.011;
                ss->repetitions[benchmark][m] = minMeasurementTime / time + 1;
            }

            ss->hostDeviceBarrier();
        }

        benchmark += 1;

        for (int stage = 0; stage < stages; stage++) {
            // host doesn't participate
            ss->hostDeviceBarrier();
            ss->hostDeviceBarrier();

            // adjust number of repetitions
            for (int m = 0; m < measurements; m++) {
                double time = ss->times[benchmark][m];
                //time = 0.011;
                ss->repetitions[benchmark][m] = minMeasurementTime / time + 1;
            }

            ss->hostDeviceBarrier();
        }

        benchmark += 1;

        CHECK(cudaStreamSynchronize(benchmarkStream));

        // print benchmark results

        int dataSizes[measurements];
        double floatDataSizes[measurements];
        for (int i = 0; i < measurements; i++) {
            dataSizes[i] = 1 << (i + 1); // +1 due to two operations: send + recv
            floatDataSizes[i] = dataSizes[i];
        }

        for (int j = 0; j < numBenchmarks; j++) {
            //for (int i = 0; i < measurements; i++) {
            //    printf("benchmark %s, dataSize %d, value %lg, bandwidth %lg\n", benchmarkName[j], dataSizes[i], ss->times[j][i], dataSizes[i] / ss->times[j][i]);
            //}
            double invThroughput, latency;
            leastSquares(measurements, floatDataSizes, ss->times[j], invThroughput, latency);
            double throughput = 1. / invThroughput;
            printf("benchmark %s latency %lg us throughput %lg GB/s\n", benchmarkName[j], latency * 1e6, throughput / 1e9);
        }

        printf("=== benchmark ===\n");
    }



    CHECK(cudaStreamDestroy(hostCopyStream));

    ss->~SharedState();
    CHECK(cudaFree(ss));
}
