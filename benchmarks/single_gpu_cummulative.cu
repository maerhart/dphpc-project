#include <new>
#include <cstdio>
#include <chrono>
#include <cassert>

#include <cuda.h>

#include "host_device_comm.cuh"

cudaStream_t hostCopyStream;
cudaStream_t benchmarkKernelStream;

const int numMeasurements = 23; 
const size_t maxBufferSize = 1 << numMeasurements; 
const int stages = 2;

struct SendRecv {
    void* volatile sendPtr;
    void* volatile recvPtr;
    volatile bool sendCompleted;
    volatile bool recvCompleted;
    volatile size_t transferSize;
};

struct SharedState {
    SharedState() {
        for (int m = 0; m < numMeasurements; m++) {
            sequentialTransferRepetitions[m] = 1;
            asyncTransferRepetitions[m] = 1;
        }
    }

    __device__ int size() {
        return blockDim.x * gridDim.x;
    }

    __device__ int rank() {
        int rank = threadIdx.x + blockDim.x * blockIdx.x;
        return rank;
    }

    __device__ void send(void* buf, size_t size, int dst) {
        SendRecv& dsr = sr[dst];
        while (dsr.sendCompleted || dsr.recvCompleted) {}
        dsr.sendPtr = buf;
        dsr.transferSize = size;
    }

    __device__ void recv(void* buf, size_t size, int src) {
        SendRecv& dsr = sr[rank()];
        while (dsr.recvCompleted || dsr.sendCompleted) {}
        dsr.recvPtr = buf;
        dsr.transferSize = size;
    }

    __device__ void waitSendFinish(int dst) {
        SendRecv& dsr = sr[dst];
        while (!dsr.sendCompleted) {}
        dsr.sendCompleted = false;
    }

    __device__ void waitRecvFinish(int src) {
        SendRecv& dsr = sr[rank()];
        while (!dsr.recvCompleted) {}
        dsr.recvCompleted = false;
    }

    __host__ __device__ void waitTransferArguments(int dst) {
        SendRecv& dsr = sr[dst];
        while (!(dsr.sendPtr && dsr.recvPtr)) {}
    }

    __host__ __device__ void finishTransfer(int dst) {
        SendRecv& dsr = sr[dst];
        dsr.sendPtr = nullptr;
        dsr.recvPtr = nullptr;
        dsr.sendCompleted = true;
        dsr.recvCompleted = true;
    }

    __device__ void sequentialTransfer(int dst) {
        SendRecv& dsr = sr[dst];
        waitTransferArguments(dst);
        memcpy(dsr.recvPtr, dsr.sendPtr, dsr.transferSize);
        finishTransfer(dst);
    }

    //__host__ __device__
    void asyncTransfer(int dst) {
        SendRecv& dsr = sr[dst];
        waitTransferArguments(dst);
//#if __CUDA_ARCH__
//        cudaMemcpyAsync(dsr.recvPtr, dsr.sendPtr, dsr.transferSize, cudaMemcpyDefault);
//        cudaDeviceSynchronize();
//#else
        CUDA_CHECK(cudaMemcpyAsync(dsr.recvPtr, dsr.sendPtr, dsr.transferSize, cudaMemcpyDefault, hostCopyStream));
        CUDA_CHECK(cudaStreamSynchronize(hostCopyStream));
//#endif
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

    long long elapsedTicks = -1;

    long long sequentialTransferResults[numMeasurements] = {0};
    int sequentialTransferRepetitions[numMeasurements] = {0};

    long long asyncTransferResults[numMeasurements] = {0};
    int asyncTransferRepetitions[numMeasurements] = {0};

    volatile bool exitting = false;

    HostDeviceComm hdc;
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

__global__ void sequentialTransferKernel(SharedState* ss) {
    long long startClock = clock64();

    char* buffer = ss->threadBuffers[ss->rank()];

    int rank = ss->rank();
    int isEven = rank % 2 == 0;
    int pairId = (rank / 2) * 2 + (isEven ? 1 : 0);

    // bechmark sequential device transfers
    if (ss->rank() == 0) {
        printf("*** Start benchmark sequential device transfers ***\n");
    }

    for (int m = 0; m < numMeasurements; m++) {
        size_t bufferSize = 1 << m;

        for (int stage = 0; stage < stages; stage++) {
            int repetitions = ss->sequentialTransferRepetitions[m];

            //cg::this_grid().sync();
            //ss->deviceBarrier();
            ss->hdc.deviceBarrier();
            long long t1 = clock64();
            //cg::this_grid().sync();
            //ss->deviceBarrier();
            ss->hdc.deviceBarrier();

            for (int r = 0; r < repetitions; r++) {
                if (isEven) {
                    ss->send(buffer, bufferSize, pairId);
                    ss->sequentialTransfer(pairId);
                    ss->waitSendFinish(pairId);

                    ss->recv(buffer, bufferSize, pairId);
                    ss->waitRecvFinish(pairId);
                } else {
                    ss->recv(buffer, bufferSize, pairId);
                    ss->waitRecvFinish(pairId);

                    ss->send(buffer, bufferSize, pairId);
                    ss->sequentialTransfer(pairId);
                    ss->waitSendFinish(pairId);
                }
                //cg::this_grid().sync();
                //ss->deviceBarrier();
                //ss->hdc.deviceBarrier();
            }

            //cg::this_grid().sync();
            //ss->deviceBarrier();
            ss->hdc.deviceBarrier();
            long long t2 = clock64();

            if (ss->rank() == 0) {
                long long measuredClocks = (t2 - t1) / repetitions;
                ss->sequentialTransferResults[m] = measuredClocks;
                long long desiredClocks = 100000000ll;
                ss->sequentialTransferRepetitions[m] = desiredClocks / measuredClocks + 1;
                printf("bufferSize %lld repetitions %d measuredClocks %lld\n", (long long)bufferSize, repetitions, measuredClocks);
            }
            //cg::this_grid().sync();
            //ss->deviceBarrier();
            ss->hdc.deviceBarrier();
        }
    }
    
    // finalize benchmark kernel

    long long endClock = clock64();
    if (ss->rank() == 0) {
        ss->elapsedTicks = endClock - startClock;
    }
}

//__global__ void asyncTransferKernel(SharedState* ss) {
//    long long startClock = clock64();
//
//    char* buffer = ss->threadBuffers[ss->rank()];
//
//    int rank = ss->rank();
//    int isEven = rank % 2 == 0;
//    int pairId = (rank / 2) * 2 + (isEven ? 1 : 0);
//    // benchmark asynchronous host transfers
//    if (ss->rank() == 0) {
//        printf("*** Start benchmark async host transfers ***\n");
//    }
//
//    memfence();
//
//    for (int m = 0; m < numMeasurements; m++) {
//        size_t bufferSize = 1 << m;
//
//        for (int stage = 0; stage < stages; stage++) {
//            int repetitions = ss->asyncTransferRepetitions[m];
//
//            //cg::this_grid().sync();
//            ss->deviceBarrier();
//            long long t1 = clock64();
//            //cg::this_grid().sync();
//            ss->deviceBarrier();
//
//            for (int r = 0; r < repetitions; r++) {
//                if (isEven) {
//                    ss->send(buffer, bufferSize, pairId);
//                    ss->asyncTransfer(pairId);
//                    ss->waitSendFinish(pairId);
//
//                    ss->recv(buffer, bufferSize, pairId);
//                    ss->waitRecvFinish(pairId);
//                } else {
//                    ss->recv(buffer, bufferSize, pairId);
//                    ss->waitRecvFinish(pairId);
//
//                    ss->send(buffer, bufferSize, pairId);
//                    ss->asyncTransfer(pairId);
//                    ss->waitSendFinish(pairId);
//                }
//                //cg::this_grid().sync();
//                ss->deviceBarrier();
//            }
//
//            //cg::this_grid().sync();
//            ss->deviceBarrier();
//            long long t2 = clock64();
//
//            if (ss->rank() == 0) {
//                long long measuredClocks = (t2 - t1) / repetitions;
//                ss->asyncTransferResults[m] = measuredClocks;
//                long long desiredClocks = 200000000ll;
//                ss->asyncTransferRepetitions[m] = desiredClocks / measuredClocks + 1;
//                printf("bufferSize %lld repetitions %d measuredClocks %lld\n", (long long)bufferSize, repetitions, measuredClocks);
//            }
//            //cg::this_grid().sync();
//            ss->deviceBarrier();
//        }
//    }
//
//    if (ss->rank() == 0) {
//        ss->exitting = true;
//    }
//
//    // finalize benchmark kernel
//
//    long long endClock = clock64();
//    if (ss->rank() == 0) {
//        ss->elapsedTicks = endClock - startClock;
//    }
//}

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

    CUDA_CHECK(cudaStreamCreate(&hostCopyStream));
    CUDA_CHECK(cudaStreamCreate(&benchmarkKernelStream));

    int gridDim = 1;
    int blockDim = 1;
    void* kernelArgs[] = { (void*) &ss };
    size_t sharedMem = 0;

    int blocksPerMP1 = -1;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerMP1, sequentialTransferKernel, blockDim, sharedMem));
    printf("Max active blocks per multiprocessor (sequential transfer kernel) %d\n", blocksPerMP1);
    //int blocksPerMP2 = -1;
    //CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerMP2, asyncTransferKernel, blockDim, sharedMem));
    //printf("Max active blocks per multiprocessor (async transfer kernel) %d\n", blocksPerMP2);
    gridDim = blocksPerMP1;
    //gridDim = blocksPerMP1 < blocksPerMP2 ? blocksPerMP1 : blocksPerMP2;

    int device = 0;
    int multiProcessorCount = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&multiProcessorCount, cudaDevAttrMultiProcessorCount, device));
    gridDim *= multiProcessorCount;
    int totalThreads = gridDim * blockDim;
    printf("Number of multiprocessors %d, total blocks %d, total threads %d\n", multiProcessorCount, gridDim, totalThreads);

    ss->allocatePointerArrays(totalThreads);
    ss->allocateThreadBuffers(totalThreads);

    // BENCHMARK 1 START
    auto time_start = hrclock::now();
    //CUDA_CHECK(cudaLaunchCooperativeKernel((void*)sequentialTransferKernel, dim3(gridDim), dim3(blockDim), kernelArgs, sharedMem, benchmarkKernelStream));
    CUDA_CHECK(cudaLaunchKernel((void*)sequentialTransferKernel, dim3(gridDim), dim3(blockDim), kernelArgs, sharedMem, benchmarkKernelStream));
    CUDA_CHECK(cudaStreamSynchronize(benchmarkKernelStream));
    auto time_end = hrclock::now();

    double benchmarkTime = nanoseconds(time_end - time_start).count();
    double clkRate = ss->elapsedTicks / benchmarkTime;
    printf("Total benchmark 1 time %lg s, GPU clock ticks %lld\n", benchmarkTime / 1e9, ss->elapsedTicks);
    printf("GPU clock rate %lg GHz\n", clkRate);
    // BENCHMARK 1 END

    for (int m = 0; m < numMeasurements; m++) {
        long long measuredClocks = ss->sequentialTransferResults[m];
        long long dataSize = 2 << m;
        double time = measuredClocks / clkRate / 1e9;
        double throughput = dataSize / time / 1e9;
        int numThreadPairs = totalThreads / 2;
        double cummulativeTrhoughput = numThreadPairs * throughput;
        int reps = ss->sequentialTransferRepetitions[m];
        printf("sequentialDeviceTransfer bytes %lld reps %d clocks %lld time %lg s bw_pair %lg GB/s bw_all %lg GB/s\n", dataSize, reps, measuredClocks, time, throughput, cummulativeTrhoughput);
    }

    // BENCHMARK 2 START
    //CUDA_CHECK(cudaLaunchCooperativeKernel((void*)asyncTransferKernel, dim3(gridDim), dim3(blockDim), kernelArgs, sharedMem, benchmarkKernelStream));
    //CUDA_CHECK(cudaLaunchKernel((void*)asyncTransferKernel, dim3(gridDim), dim3(blockDim), kernelArgs, sharedMem, benchmarkKernelStream));

    //// host-side copies during benchmark execution
    //while (!ss->exitting) {
    //    for (int thread = 0; thread < totalThreads; thread++) {
    //        SendRecv& dsr = ss->sr[thread];
    //        if (dsr.sendPtr && dsr.recvPtr) {
    //            ss->asyncTransfer(thread);
    //        }
    //    }
    //}

    //CUDA_CHECK(cudaStreamSynchronize(benchmarkKernelStream));
    // BENCHMARK 2 END

    //for (int m = 0; m < numMeasurements; m++) {
    //    long long measuredClocks = ss->asyncTransferResults[m];
    //    long long dataSize = 2 << m;
    //    double time = measuredClocks / clkRate / 1e9;
    //    double throughput = dataSize / time / 1e9;
    //    int numThreadPairs = totalThreads / 2;
    //    double cummulativeTrhoughput = numThreadPairs * throughput;
    //    int reps = ss->asyncTransferRepetitions[m];
    //    printf("asyncHostTransfer bytes %lld reps %d clocks %lld time %lg s bw_pair %lg GB/s bw_all %lg GB/s\n", dataSize, reps, measuredClocks, time, throughput, cummulativeTrhoughput);
    //}

    CUDA_CHECK(cudaStreamDestroy(benchmarkKernelStream));
    CUDA_CHECK(cudaStreamDestroy(hostCopyStream));

    ss->~SharedState();
    CUDA_CHECK(cudaFree(ss));
}
