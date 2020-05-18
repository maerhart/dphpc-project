#ifndef TEST_RUNNER_CUH
#define TEST_RUNNER_CUH

#include "cuda_mpi.cuh"

#include "libc_processor.cuh"

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
    {
        mSharedStateContext.numThreads = numThreads;
        
        int device = 0;
        int peakClockKHz;
        CUDA_CHECK(cudaDeviceGetAttribute(&peakClockKHz, cudaDevAttrClockRate, device));
        mThreadPrivateStateContext.peakClockKHz = peakClockKHz;
    }

    template <typename KernelClass>
    void run() {
        
        mSharedStateContext.numThreads = mSharedStateContext.numThreads;
        CudaMPI::SharedState::Holder sharedStateHolder(mSharedStateContext);

        bool* ok;
        CUDA_CHECK(cudaMallocManaged(&ok, sizeof(bool) * mSharedStateContext.numThreads));
        for (int i = 0; i < mSharedStateContext.numThreads; i++) {
            ok[i] = false;
        }

        CudaMPI::SharedState* sharedState = sharedStateHolder.get();

        void* params[] = {
            (void*)&sharedState,
            (void*)&mThreadPrivateStateContext,
            (void*)&ok
        };

        CUDA_CHECK(cudaLaunchCooperativeKernel((void*)testRunnerKernel<KernelClass>, mSharedStateContext.numThreads, 1, params));
        CUDA_CHECK(cudaPeekAtLastError());
        
        std::set<int> unfinishedThreads;
        for (int i = 0; i < mSharedStateContext.numThreads; i++) {
            unfinishedThreads.insert(i);
        }

        while (!unfinishedThreads.empty()) {
            sharedState->deviceToHostCommunicator.processIncomingMessages([&](void* ptr, size_t size, int threadRank) {
                if (ptr == 0 && size == 0) {
                    int erased = unfinishedThreads.erase(threadRank);
                    assert(erased);
                } else {
                    process_gpu_libc(ptr, size);
                }
            });
        }
        
        CUDA_CHECK(cudaDeviceSynchronize());

        for (int i = 0; i < mSharedStateContext.numThreads; i++) {
            REQUIRE(ok[i] == true);
        }
    }
    
    CudaMPI::SharedState::Context mSharedStateContext;
    CudaMPI::ThreadPrivateState::Context mThreadPrivateStateContext;
};

#endif
