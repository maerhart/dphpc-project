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

static const int measurements = 23; // 8 MB
const size_t maxBufferSize = 1 << measurements; 
const int numBenchmarks = 5;
const char* benchmarkName[5] = {
    "sender_issues_transfer",
    "host_issues_tranfer_memcpy",
    "host_issues_tranfer_copy_kernel",
    "device_issues_tranfer_memcpy",
    "device_issues_tranfer_copy_kernel"
};

cudaStream_t hostCopyStream;
cudaStream_t benchmarkStream;

struct SharedState {
    SharedState() {
        CHECK(cudaMalloc(&buffer1, maxBufferSize));
        CHECK(cudaMalloc(&buffer2, maxBufferSize));

        CHECK(cudaMemset(buffer1, 0, maxBufferSize));
        CHECK(cudaMemset(buffer2, 0, maxBufferSize));
    }

    // clock rate related 
    const int warmup = 10;
    const float interval = 0.1;
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

    double times[numBenchmarks][measurements];
    
    __device__ int rank() {
        return threadIdx.x + blockDim.x * blockIdx.x;
    }

    __device__ void send(void* buf, size_t size) {
        __threadfence_system();
        while (sendPtr) {}
        sendPtr = buf;
        transferSize = size;
    }

    __device__ void recv(void* buf, size_t size) {
        __threadfence_system();
        while (recvPtr) {}
        recvPtr = buf;
        transferSize = size;
    }

    __host__ __device__ void waitTransferFinish() {
#if __CUDA_ARCH__
        __threadfence_system();
#else
        __sync_synchronize();
#endif
        while (sendPtr || recvPtr) {}
    }

    __host__ __device__ void waitTransferArguments() {
#if __CUDA_ARCH__
        __threadfence_system();
#else
        __sync_synchronize();
#endif
        while (!sendPtr || !recvPtr) {}
    }

    __host__ __device__ void finishTransfer() {
        sendPtr = nullptr;
        recvPtr = nullptr;
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

};

__global__ void clockMeasurementKernel(SharedState* ss) {
    volatile bool& cpuReady = ss->clockMeasurementCPUReady;
    volatile bool& gpuReady = ss->clockMeasurementGPUReady;

    for (int i = 0; i < ss->warmup; i++) {
        gpuReady = true;
        __threadfence_system();
        while (!cpuReady) {}
        long long t1 = clock64();
        __threadfence_system();
        while (cpuReady) {}
        long long t2 = clock64();
        gpuReady = false;

        printf("%d clocks per %lg seconds\n", t2 - t1, ss->interval);
        ss->clocksPerSecond = (t2 - t1) / ss->interval; // use value from last iteration
    }
}

__global__ void checkClockRateCalibration(SharedState* ss) {
    // check clock rate calibration
    long long t1 = clock64();
    __threadfence_system();
    while (ss->dt(t1, clock64()) < ss->interval) {}
    long long t2 = clock64();
    double dt = ss->dt(t1, t2);
    printf("dt on GPU = %lg s\n", dt);
}

__device__ void startBenchmark(SharedState* ss) {
    if (ss->rank() == 0) {
        ss->thread1ready = true;
    } else {
        ss->thread2ready = true;
    }

    __threadfence_system();
    while (!ss->cpuReady) {}
}

__device__ void finishBenchmark(SharedState* ss) {
    if (ss->rank() == 0) {
        ss->thread1ready = false;
    } else {
        ss->thread2ready = false;
    }

    __threadfence_system();
    while (ss->cpuReady) {}
}

__device__ void compute_latency_and_throughput(int n, const double* x, const double* y, double* a, double* b) {
    
}

__global__ void benchmarkKernel(SharedState* ss) {
    int rank = ss->rank();
    double times[measurements];

    // benchmark 1: sender issues transfer
    startBenchmark(ss);   

    for (int m = 0; m < measurements; m++) {
        size_t bufferSize = 1 << m;
        long long t1 = clock64();
        if (rank == 0) {
            ss->send(ss->buffer1, bufferSize);
            ss->deviceSequentialTransfer();

            ss->recv(ss->buffer1, bufferSize);
            ss->waitTransferFinish();
        } else if (rank == 1) {
            ss->recv(ss->buffer2, bufferSize);
            ss->waitTransferFinish();

            ss->send(ss->buffer2, bufferSize);
            ss->deviceSequentialTransfer();
        }
        long long t2 = clock64();
        times[m] = ss->dt(t1, t2);
        //if (rank == 0) {
        //    printf("sender_issues_tranfer bufferSize = %d, time = %f, bw = %f\n", bufferSize, times[m], bufferSize / times[m]);
        //}
    }

    // copy measurements to the host
    if (rank == 0) {
        memcpy(ss->times[0], times, sizeof(double) * measurements);
    }

    finishBenchmark(ss);

    // benchmark 2: host issues transfer with memcpy

    startBenchmark(ss);

    for (int m = 0; m < measurements; m++) {
        size_t bufferSize = 1 << m;
        long long t1 = clock64();
        if (rank == 0) {
            ss->send(ss->buffer1, bufferSize);
            ss->waitTransferFinish();

            ss->recv(ss->buffer1, bufferSize);
            ss->waitTransferFinish();
        } else if (rank == 1) {
            ss->recv(ss->buffer2, bufferSize);
            ss->waitTransferFinish();

            ss->send(ss->buffer2, bufferSize);
            ss->waitTransferFinish();
        }
        long long t2 = clock64();
        times[m] = ss->dt(t1, t2);
        //if (rank == 0) {
        //    printf("host_issues_tranfer_memcpy bufferSize = %d, time = %f, bw = %f\n", bufferSize, times[m], bufferSize / times[m]);
        //}
    }

    // copy measurements to the host
    if (rank == 0) {
        memcpy(ss->times[1], times, sizeof(double) * measurements);
    }

    finishBenchmark(ss);

    // benchmark 3: host issues transfer copy kernel

    startBenchmark(ss);

    for (int m = 0; m < measurements; m++) {
        size_t bufferSize = 1 << m;
        long long t1 = clock64();
        if (rank == 0) {
            ss->send(ss->buffer1, bufferSize);
            ss->waitTransferFinish();

            ss->recv(ss->buffer1, bufferSize);
            ss->waitTransferFinish();
        } else if (rank == 1) {
            ss->recv(ss->buffer2, bufferSize);
            ss->waitTransferFinish();

            ss->send(ss->buffer2, bufferSize);
            ss->waitTransferFinish();
        }
        long long t2 = clock64();
        times[m] = ss->dt(t1, t2);
        //if (rank == 0) {
        //    printf("host_issues_tranfer_copy_kernel bufferSize = %d, time = %f, bw = %f\n", bufferSize, times[m], bufferSize / times[m]);
        //}
    }

    // copy measurements to the host
    if (rank == 0) {
        memcpy(ss->times[2], times, sizeof(double) * measurements);
    }

    finishBenchmark(ss);

    // benchmark 4: device issues transfer memcpy
    startBenchmark(ss);

    for (int m = 0; m < measurements; m++) {
        size_t bufferSize = 1 << m;
        long long t1 = clock64();
        if (rank == 0) {
            ss->send(ss->buffer1, bufferSize);
            ss->memcpyTransfer();

            ss->recv(ss->buffer1, bufferSize);
            ss->waitTransferFinish();
        } else if (rank == 1) {
            ss->recv(ss->buffer2, bufferSize);
            ss->waitTransferFinish();

            ss->send(ss->buffer2, bufferSize);
            ss->memcpyTransfer();
        }
        long long t2 = clock64();
        times[m] = ss->dt(t1, t2);
        //if (rank == 0) {
        //    printf("device_issues_tranfer_memcpy bufferSize = %d, time = %f, bw = %f\n", bufferSize, times[m], bufferSize / times[m]);
        //}
    }

    // copy measurements to the host
    if (rank == 0) {
        memcpy(ss->times[3], times, sizeof(double) * measurements);
    }

    finishBenchmark(ss);

    // benchmark 5: device issues transfer copy kernel
    startBenchmark(ss);

    for (int m = 0; m < measurements; m++) {
        size_t bufferSize = 1 << m;
        long long t1 = clock64();
        if (rank == 0) {
            ss->send(ss->buffer1, bufferSize);
            ss->copyKernelTransfer();

            ss->recv(ss->buffer1, bufferSize);
            ss->waitTransferFinish();
        } else if (rank == 1) {
            ss->recv(ss->buffer2, bufferSize);
            ss->waitTransferFinish();

            ss->send(ss->buffer2, bufferSize);
            ss->copyKernelTransfer();
        }
        long long t2 = clock64();
        times[m] = ss->dt(t1, t2);
        //if (rank == 0) {
        //    printf("device_issues_tranfer_copy_kernel bufferSize = %d, time = %f, bw = %f\n", bufferSize, times[m], bufferSize / times[m]);
        //}
    }

    // copy measurements to the host
    if (rank == 0) {
        memcpy(ss->times[4], times, sizeof(double) * measurements);
    }

    finishBenchmark(ss);
}

using hrclock = std::chrono::high_resolution_clock;

template <typename T>
auto nanoseconds(T x) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(x);
}

void hostStartBenchmark(SharedState* ss) {
    __sync_synchronize();
    while (!ss->thread1ready || !ss->thread2ready) {}
    ss->cpuReady = true;
}
void hostFinishBenchmark(SharedState* ss) {
    __sync_synchronize();
    while (ss->thread1ready || ss->thread2ready) {}
    ss->cpuReady = false;
}

int main() {
    SharedState* ss = nullptr;
    CHECK(cudaMallocManaged(&ss, sizeof(SharedState)));
    ss = new (ss) SharedState;

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

        for (int i = 0; i < ss->warmup; i++) {
            __sync_synchronize();
            while (!gpuReady) {} // wait GPU to start
            auto t1 = hrclock::now();
            cpuReady = true;
            __sync_synchronize();
            while (nanoseconds(hrclock::now() - t1).count() / 1e9 < ss->interval) {}
            cpuReady = false;
            __sync_synchronize();
            while (gpuReady) {} // wait GPU to finish
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

        {
            hostStartBenchmark(ss);
            // host doesn't participate
            hostFinishBenchmark(ss);
        }

        {
            hostStartBenchmark(ss);
            for (int m = 0; m < measurements; m++) {
                ss->memcpyTransfer();
                ss->memcpyTransfer();
            }
            hostFinishBenchmark(ss);
        }

        {
            hostStartBenchmark(ss);
            for (int m = 0; m < measurements; m++) {
                ss->memcpyTransfer();
                ss->memcpyTransfer();
            }
            hostFinishBenchmark(ss);
        }

        {
            hostStartBenchmark(ss);
            // host doesn't participate
            hostFinishBenchmark(ss);
        }

        {
            hostStartBenchmark(ss);
            // host doesn't participate
            hostFinishBenchmark(ss);
        }

        CHECK(cudaStreamSynchronize(benchmarkStream));

        // print benchmark results

        int dataSizes[measurements];
        for (int i = 0; i < measurements; i++) {
            dataSizes[i] = 1 << (i + 1); // +1 due to two operations: send + recv
        }

        for (int j = 0; j < numBenchmarks; j++) {
            for (int i = 0; i < measurements; i++) {
                printf("benchmark %s, dataSize %d, value %lg, bandwidth %lg\n", benchmarkName[j], dataSizes[i], ss->times[j][i], dataSizes[i] / ss->times[j][i]);
            }

        }

        printf("=== benchmark ===\n");
    }



    CHECK(cudaStreamDestroy(hostCopyStream));

    ss->~SharedState();
    CHECK(cudaFree(ss));
}
