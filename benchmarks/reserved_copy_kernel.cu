#include <new>
#include <cstdio>
#include <chrono>
#include <cassert>

#include <cuda.h>

#include "host_device_comm.cuh"

cudaStream_t benchmarkKernelStream;

const int numMeasurements = 23; 
const size_t maxBufferSize = 1 << numMeasurements; 
const int stages = 2;


__forceinline__ __device__
void copyKernel(void* dst, void* src, size_t size, size_t start, size_t step) {
    char* d = (char*) dst;
    char* s = (char*) src;
    
    for (size_t i = start; i < size; i += step) {
        d[i] = s[i];
    }
}

struct SendRecv {
    void* volatile sendPtr;
    void* volatile recvPtr;
    volatile bool sendCompleted;
    volatile bool recvCompleted;
    volatile size_t transferSize;
    
    __host__ __device__ void finishTransfer() {
//         printf("Finish transfer %p\n", this);
        sendPtr = nullptr;
        recvPtr = nullptr;
        sendCompleted = true;
        recvCompleted = true;
        transferSize = 0;
    }
};

struct SharedState {
    SharedState() {
        for (int m = 0; m < numMeasurements; m++) {
            sequentialTransferRepetitions[m] = 1;
            asyncTransferRepetitions[m] = 1;
        }
    }
    
    __device__ bool isWorkload() {
        return blockIdx.x < workloadBlocks;
    }
    
    __device__ int workloadSize() {
        return workloadBlocks * blockDim.x;
    }
    
    __device__ int workloadRank() {
        return threadIdx.x + blockDim.x * blockIdx.x;
    }
    
    __device__ int copySize() {
        return copyBlocks * blockDim.x;
    }
    
    __device__ int copyRank() {
        return threadIdx.x + blockDim.x * (blockIdx.x - workloadBlocks);
    }

    __device__ void send(void* buf, size_t size, int dst) {
        SendRecv& dsr = sr[dst];
        WAIT(!dsr.sendCompleted && !dsr.recvCompleted);
        dsr.transferSize = size;
        dsr.sendPtr = buf;
    }

    __device__ void recv(void* buf, size_t size, int src) {
        int myrank = blockIdx.x; // only relevant for benchmark
        SendRecv& dsr = sr[myrank];
        WAIT(!dsr.recvCompleted && !dsr.sendCompleted);
        dsr.transferSize = size;
        dsr.recvPtr = buf;
        
        // put copy requiest in queue
        WAIT(dsr.sendPtr);
        mq.push(&dsr);
    }

    __device__ void waitSendFinish(int dst) {
        SendRecv& dsr = sr[dst];
//         printf("[sender] Waiting transfer %p to finish\n", &dsr);
        WAIT(dsr.sendCompleted);
        dsr.sendCompleted = false;
    }

    __device__ void waitRecvFinish(int src) {
        int myrank = blockIdx.x; // only relevant for benchmark
        SendRecv& dsr = sr[myrank];
//         printf("[receiver] Waiting transfer %p to finish\n", &dsr);
        WAIT(dsr.recvCompleted);
        dsr.recvCompleted = false;
    }

    __host__ __device__ void waitTransferArguments(int dst) {
        SendRecv& dsr = sr[dst];
        WAIT(dsr.sendPtr && dsr.recvPtr);
    }

    __host__ __device__ void finishTransfer(int dst) {
        SendRecv& dsr = sr[dst];
        dsr.finishTransfer();
    }

    __device__ void sequentialTransfer(int dst) {
        SendRecv& dsr = sr[dst];
        waitTransferArguments(dst);
        memcpy(dsr.recvPtr, dsr.sendPtr, dsr.transferSize);
        finishTransfer(dst);
    }

    SendRecv* sr;
    void allocatePointerArrays(int totalThreads) {
        CUDA_CHECK(cudaMallocManaged(&sr, totalThreads * sizeof(SendRecv)));
        CUDA_CHECK(cudaMemset(sr, 0, totalThreads * sizeof(SendRecv)));
    }

    char** threadBuffers;
    void allocateThreadBuffers(int totalThreads) {
        CUDA_CHECK(cudaMallocManaged(&threadBuffers, totalThreads * sizeof(char*)));
        for (int i = 0; i < totalThreads; i++) {
            CUDA_CHECK(cudaMalloc(&threadBuffers[i], maxBufferSize));
        }
    }
    
    // threads that exitted from benchmark
    // used to stop copyImpl when all threads are exitted
    int finishedThreads = 0; 
    
    volatile unsigned int benchmarkBarrier = 0;
    volatile unsigned int copyBarrier = 0;
    
    int copyBlocks = -1;
    int workloadBlocks = -1;

    long long elapsedTicks = -1;

    long long sequentialTransferResults[numMeasurements] = {0};
    int sequentialTransferRepetitions[numMeasurements] = {0};

    long long asyncTransferResults[numMeasurements] = {0};
    int asyncTransferRepetitions[numMeasurements] = {0};

    HostDeviceComm hdc;
    
    SendRecv* volatile nextSR = nullptr;
    
    Queue<SendRecv*> mq = {4};
};

// https://en.wikipedia.org/wiki/Ordinary_least_squares#Simple_linear_regression_model
void leastSquares(int n, const double* x, const double* y, double& a, double& b) {
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
}

__device__ void benchmarkImpl(SharedState* ss) {
    auto startClock = clock64();
    
    // early exit from unused threads
    // we use only first threads of each block in this benchmark
    int myrank = blockIdx.x;
    int mysize = ss->workloadSize() / blockDim.x;
    if (threadIdx.x != 0) return;

    if (myrank == 0) {
        printf("Benchmark started\n");
    }
    
    char* buffer = ss->threadBuffers[myrank];

    int isEven = myrank % 2 == 0;
    int pairId = (myrank / 2) * 2 + (isEven ? 1 : 0);

    for (int m = 0; m < numMeasurements; m++) {
        size_t bufferSize = 1 << m;

        for (int stage = 0; stage < stages; stage++) {
            int repetitions = ss->asyncTransferRepetitions[m];

            myCudaBarrier(mysize, &ss->benchmarkBarrier, myrank == 0);
            auto t1 = clock64();
            myCudaBarrier(mysize, &ss->benchmarkBarrier, myrank == 0);

            for (int r = 0; r < repetitions; r++) {
//                 printf("repeat %d, rank %d\n", r, myrank);
                if (isEven) {
                    ss->send(buffer, bufferSize, pairId);
                    ss->waitSendFinish(pairId);

                    ss->recv(buffer, bufferSize, pairId);
                    ss->waitRecvFinish(pairId);
                } else {
                    ss->recv(buffer, bufferSize, pairId);
                    ss->waitRecvFinish(pairId);

                    ss->send(buffer, bufferSize, pairId);
                    ss->waitSendFinish(pairId);
                }
//                 myCudaBarrier(mysize, &ss->benchmarkBarrier, myrank == 0); // TODO: remove, only for debugging
            }

            myCudaBarrier(mysize, &ss->benchmarkBarrier, myrank == 0);

            auto t2 = clock64();

            if (myrank == 0) {
                long long measuredClocks = (t2 - t1) / repetitions;
                ss->asyncTransferResults[m] = measuredClocks;
                long long desiredClocks = 200000000ll;
                ss->asyncTransferRepetitions[m] = desiredClocks / measuredClocks + 1;
//                 ss->asyncTransferRepetitions[m] = 1;
                printf("bufferSize %lld repetitions %d measuredClocks %lld\n", (long long)bufferSize, repetitions, measuredClocks);
            }

            myCudaBarrier(mysize, &ss->benchmarkBarrier, myrank == 0);
        }
    }

    // finalize benchmark kernel

    auto endClock = clock64();
    if (myrank == 0) {
        ss->elapsedTicks = endClock - startClock;
    }
}

__forceinline__
__device__ bool copyImpl(SharedState* ss) {
    int rank = ss->copyRank();
    int size = ss->copySize();
    int workloadBlocks = ss->workloadSize() / blockDim.x;
    
    if (rank == 0) {
        SendRecv* sr = nullptr;
        if (ss->mq.tryPop(sr)) {
            ss->nextSR = sr;
        } else {
            ss->nextSR = nullptr;
        }
        
    }
        
    myCudaBarrier(size, &ss->copyBarrier, rank == 0);
            
    if (!ss->nextSR) return false;
        
    SendRecv sr = *ss->nextSR;

    copyKernel(sr.recvPtr, sr.sendPtr, sr.transferSize, rank, size);
      
    myCudaBarrier(size, &ss->copyBarrier, rank == 0);
        
    if (rank == 0) {
        ss->nextSR->finishTransfer();
    }
    
    return true;
}

__global__ void benchmarkKernel(SharedState* ss) {    
    if (ss->isWorkload()) {
        benchmarkImpl(ss);
        atomicAdd(&ss->finishedThreads, 1);
    } else {
        while (volatileAccess(ss->finishedThreads) != ss->workloadSize()) {
            copyImpl(ss);
        }
    }
}

using hrclock = std::chrono::high_resolution_clock;

template <typename T>
auto nanoseconds(T x) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(x);
}

int main() {
    SharedState* ss = nullptr;
    CUDA_CHECK(cudaMallocManaged(&ss, sizeof(SharedState)));
    ss = new (ss) SharedState;

    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaStreamCreate(&benchmarkKernelStream));

    
    int blockDim = 1;
    blockDim = 1;
    void* kernelArgs[] = { (void*) &ss };
    size_t sharedMem = 0;

    int blocksPerMP = -1;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerMP, benchmarkKernel, blockDim, sharedMem));
    printf("Max active blocks per multiprocessor (async transfer kernel) %d\n", blocksPerMP);

    int device = 0;
    int multiProcessorCount = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&multiProcessorCount, cudaDevAttrMultiProcessorCount, device));
    int gridDim = blocksPerMP * multiProcessorCount;
    //gridDim = 3;
    int totalThreads = gridDim * blockDim;
    printf("Number of multiprocessors %d, total blocks %d, total threads %d\n", multiProcessorCount, gridDim, totalThreads);

    ss->copyBlocks = gridDim * 5 / 10; // % of thread blocks for copies
    //ss->copyBlocks = 1;
    ss->workloadBlocks = gridDim - ss->copyBlocks; // remaining threads for MPI processes
    //ss->workloadBlocks = 2;
    
    printf("copyBlocks %d workloadBlocks %d\n", ss->copyBlocks, ss->workloadBlocks);
    
    ss->allocatePointerArrays(ss->workloadBlocks);
    ss->allocateThreadBuffers(ss->workloadBlocks);

    // BENCHMARK START
    auto time_start = hrclock::now();
    CUDA_CHECK(cudaLaunchCooperativeKernel((void*)benchmarkKernel, dim3(gridDim), dim3(blockDim), kernelArgs, sharedMem, benchmarkKernelStream));

    CUDA_CHECK(cudaStreamSynchronize(benchmarkKernelStream));
    auto time_end = hrclock::now();
    double benchmarkTime = nanoseconds(time_end - time_start).count();
    double clkRate = ss->elapsedTicks / benchmarkTime;
    printf("Total benchmark time %lg s, GPU clock ticks %lld\n", benchmarkTime / 1e9, ss->elapsedTicks);
    printf("GPU clock rate %lg GHz\n", clkRate);
    // BENCHMARK END

    for (int m = 0; m < numMeasurements; m++) {
        long long measuredClocks = ss->asyncTransferResults[m];
        long long dataSize = 2 << m;
        double time = measuredClocks / clkRate / 1e9;
        double throughput = dataSize / time / 1e9;
        int numThreadPairs = ss->workloadBlocks / 2;
        double cummulativeTrhoughput = numThreadPairs * throughput;
        int reps = ss->asyncTransferRepetitions[m];
        printf("bytes %lld reps %d clocks %lld time %lg s bw_pair %lg GB/s bw_all %lg GB/s\n", dataSize, reps, measuredClocks, time, throughput, cummulativeTrhoughput);
    }

    CUDA_CHECK(cudaStreamDestroy(benchmarkKernelStream));

    ss->~SharedState();
    CUDA_CHECK(cudaFree(ss));
}
