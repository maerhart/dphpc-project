#ifndef TEST_RUNNER_CUH
#define TEST_RUNNER_CUH

#include "cuda_mpi.cuh"

#define CATCH_CONFIG_MAIN
#include "catch.hpp"

template <typename F>
__global__ void testRunnerKernel(
    CudaMPI::SharedState* sharedState,
    CudaMPI::ThreadPrivateState::Context threadPrivateStateContext,
    bool* allOk)
{
    CudaMPI::setSharedState(sharedState);
    CudaMPI::ThreadPrivateState::Holder threadPrivateStateHolder(threadPrivateStateContext);

    int rank = cg::this_grid().thread_rank();
    bool& ok = allOk[rank];

    F::run(ok);
}

class TestRunner {
public:
    TestRunner(int numThreads)
        : mNumThreads(numThreads)
        {}

    template <typename KernelClass>
    void run() {
        CudaMPI::SharedState::Context sharedStateContext;
        sharedStateContext.numThreads = mNumThreads;
        CudaMPI::SharedState::Holder sharedStateHolder(sharedStateContext);

        int device = 0;

        CudaMPI::ThreadPrivateState::Context threadPrivateStateContext;
        int peakClockKHz;
        CUDA_CHECK(cudaDeviceGetAttribute(&peakClockKHz, cudaDevAttrClockRate, device));
        threadPrivateStateContext.peakClockKHz = peakClockKHz;

        bool* ok;
        CUDA_CHECK(cudaMallocManaged(&ok, sizeof(bool) * mNumThreads));
        for (int i = 0; i < mNumThreads; i++) {
            ok[i] = false;
        }

        CudaMPI::SharedState* sharedStatePtr = sharedStateHolder.get();

        void* params[] = {
            (void*)&sharedStatePtr,
            (void*)&threadPrivateStateContext,
            (void*)&ok
        };

        CUDA_CHECK(cudaLaunchCooperativeKernel((void*)testRunnerKernel<KernelClass>, mNumThreads, 1, params));
        CUDA_CHECK(cudaPeekAtLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        for (int i = 0; i < mNumThreads; i++) {
            REQUIRE(ok[i] == true);
        }
    }
private:
    int mNumThreads;
};

#endif
