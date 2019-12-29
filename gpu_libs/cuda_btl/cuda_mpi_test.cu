#include "cuda_mpi.cuh"

#define CATCH_CONFIG_MAIN
#include "catch.hpp"

namespace cg = cooperative_groups;


__global__ void single_int_kernel(
    CudaMPI::SharedState* sharedState,
    CudaMPI::ThreadPrivateState::Context threadPrivateStateContext,
    int* res)
{
    CudaMPI::setSharedState(sharedState);
    CudaMPI::ThreadPrivateState::Holder threadPrivateStateHolder(threadPrivateStateContext);

    if (cg::this_grid().thread_rank() == 0) {
        int x = 3456;

        CudaMPI::PendingOperation* op = CudaMPI::isend(1, &x, sizeof(int), 0, 15);

        CudaMPI::wait(op);
    } else if (cg::this_grid().thread_rank() == 1) {
        int x = 0;

        CudaMPI::PendingOperation* op = CudaMPI::irecv(0, &x, sizeof(int), 0, 15);

        CudaMPI::wait(op);

        *res = x;
    }
}

TEST_CASE("Transfer single integer", "[single_int]") {
    CudaMPI::SharedState::Context sharedStateContext = {2, 10, 10, 10, 10};
    CudaMPI::SharedState::Holder sharedStateHolder(sharedStateContext);
    CudaMPI::ThreadPrivateState::Context threadPrivateStateContext = {20};

    int* res;
    CUDA_CHECK(cudaMallocManaged(&res, sizeof(int)));
    *res = 0;

        CudaMPI::SharedState* sharedStatePtr = sharedStateHolder.get();
    
    void* params[] = {
        (void*)&sharedStatePtr, 
        (void*)&threadPrivateStateContext,
        (void*)&res
    };
    
    CUDA_CHECK(cudaLaunchCooperativeKernel((void*)single_int_kernel, 3, 1, params));
    CUDA_CHECK(cudaPeekAtLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    REQUIRE(*res == 3456);
}


__global__ void transfer_array_kernel(
    CudaMPI::SharedState* sharedState,
    CudaMPI::ThreadPrivateState::Context threadPrivateStateContext,
    bool* ok)
{
    CudaMPI::setSharedState(sharedState);
    CudaMPI::ThreadPrivateState::Holder threadPrivateStateHolder(threadPrivateStateContext);

    if (cg::this_grid().thread_rank() == 0) {
        int x[16] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};

        CudaMPI::PendingOperation* op = CudaMPI::isend(1, &x, sizeof(x), 0, 15);

        CudaMPI::wait(op);
    } else if (cg::this_grid().thread_rank() == 1) {
        int x[16] = {};

        CudaMPI::PendingOperation* op = CudaMPI::irecv(0, &x, sizeof(x), 0, 15);

        CudaMPI::wait(op);

        *ok = true;
        for (int i = 0; i < 16; i++) {
            if (x[i] != i) *ok = false;
        }
    }
}

TEST_CASE("Transfer array", "[array]") {
    CudaMPI::SharedState::Context sharedStateContext = {2, 10, 10, 10, 10};
    CudaMPI::SharedState::Holder sharedStateHolder(sharedStateContext);
    CudaMPI::ThreadPrivateState::Context threadPrivateStateContext = {20};

    bool* ok;
    CUDA_CHECK(cudaMallocManaged(&ok, sizeof(bool)));
    *ok = 0;

    CudaMPI::SharedState* sharedStatePtr = sharedStateHolder.get();
    
    void* params[] = {
        (void*)&sharedStatePtr, 
        (void*)&threadPrivateStateContext,
        (void*)&ok
    };
    
    CUDA_CHECK(cudaLaunchCooperativeKernel((void*)transfer_array_kernel, 3, 1, params));
    CUDA_CHECK(cudaPeekAtLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    REQUIRE(*ok == true);
}

__global__ void send_recv_kernel(
    CudaMPI::SharedState* sharedState,
    CudaMPI::ThreadPrivateState::Context threadPrivateStateContext,
    bool* ok)
{
    CudaMPI::setSharedState(sharedState);
    CudaMPI::ThreadPrivateState::Holder threadPrivateStateHolder(threadPrivateStateContext);

    if (cg::this_grid().thread_rank() == 0) {
        int x[16] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
        int y[16] = {};

        CudaMPI::PendingOperation* op[2];
        op[0] = CudaMPI::irecv(1, &y, sizeof(y), 0, 15);
        op[1] = CudaMPI::isend(1, &x, sizeof(x), 0, 15);

        CudaMPI::wait(op[0]);

        ok[0] = true;
        for (int i = 0; i < 16; i++) {
            if (y[i] != -i) {
                printf("thread %d, y[i] = %d\n", cg::this_grid().thread_rank(), y[i]);
                ok[0] = false;
            }
        }

        CudaMPI::wait(op[1]);
    } else if (cg::this_grid().thread_rank() == 1) {
        int x[16] = {0, -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15};
        int y[16] = {};

        CudaMPI::PendingOperation* op[2];
        op[0] = CudaMPI::irecv(0, &y, sizeof(y), 0, 15);
        op[1] = CudaMPI::isend(0, &x, sizeof(x), 0, 15);

        CudaMPI::wait(op[0]);

        ok[1] = true;
        for (int i = 0; i < 16; i++) {
            if (y[i] != i) {
                printf("thread %d, y[i] = %d\n", cg::this_grid().thread_rank(), y[i]);
                ok[1] = false;
            }
        }

        CudaMPI::wait(op[1]);
    }
}

TEST_CASE("Send receive", "[send_recv]") {
    CudaMPI::SharedState::Context sharedStateContext = {2, 10, 10, 10, 10};
    CudaMPI::SharedState::Holder sharedStateHolder(sharedStateContext);
    CudaMPI::ThreadPrivateState::Context threadPrivateStateContext = {20};

    bool* ok;
    CUDA_CHECK(cudaMallocManaged(&ok, 2 * sizeof(bool)));
    ok[0] = false;
    ok[1] = false;

    CudaMPI::SharedState* sharedStatePtr = sharedStateHolder.get();
    
    void* params[] = {
        (void*)&sharedStatePtr, 
        (void*)&threadPrivateStateContext,
        (void*)&ok
    };
    
    CUDA_CHECK(cudaLaunchCooperativeKernel((void*)send_recv_kernel, 3, 1, params));
    CUDA_CHECK(cudaPeekAtLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    REQUIRE(ok[0] == true);
    REQUIRE(ok[1] == true);
}

__global__ void repeat_sendrecv_kernel(
    CudaMPI::SharedState* sharedState,
    CudaMPI::ThreadPrivateState::Context threadPrivateStateContext,
    bool* ok)
{
    CudaMPI::setSharedState(sharedState);
    CudaMPI::ThreadPrivateState::Holder threadPrivateStateHolder(threadPrivateStateContext);

    const int numRanks = 2;
    
    int thisRank = cg::this_grid().thread_rank();
    int otherRank = (thisRank + 1) % numRanks;
    
    const int numRepeats = 5;
    const int dataSize = 1 << (numRepeats - 1);
    
    int localData[dataSize] = {};
    for (int i = 0; i < dataSize; i++) {
        localData[i] = (thisRank + 1) * i;
    }
    
    int remoteData[dataSize] = {};
    
    CudaMPI::PendingOperation* send_op[numRepeats];
    CudaMPI::PendingOperation* recv_op[numRepeats];
    
    for (int i = 0; i < numRepeats; i++) {
        int tag = i + 10;
        send_op[i] = CudaMPI::isend(otherRank, localData, sizeof(int) * (1 << i), 0, tag);
        recv_op[i] = CudaMPI::irecv(otherRank, remoteData, sizeof(int) * (1 << i), 0, tag);
    }
    
    for (int i = 0; i < numRepeats; i++) {
        CudaMPI::wait(send_op[i]);
        CudaMPI::wait(recv_op[i]);
    }
    
    ok[thisRank] = true;
    for (int i = 0; i < dataSize; i++) {
        if (remoteData[i] != (otherRank + 1) * i) ok[thisRank] = false;
    }
}

TEST_CASE("Repeat send recv", "[repeat_sendrecv]") {
    CudaMPI::SharedState::Context sharedStateContext = {2, 10, 10, 10, 10};
    CudaMPI::SharedState::Holder sharedStateHolder(sharedStateContext);
    CudaMPI::ThreadPrivateState::Context threadPrivateStateContext = {20};

    bool* ok;
    CUDA_CHECK(cudaMallocManaged(&ok, 2 * sizeof(bool)));
    ok[0] = false;
    ok[1] = false;

    CudaMPI::SharedState* sharedStatePtr = sharedStateHolder.get();
    
    void* params[] = {
        (void*)&sharedStatePtr, 
        (void*)&threadPrivateStateContext,
        (void*)&ok
    };
    
    CUDA_CHECK(cudaLaunchCooperativeKernel((void*)repeat_sendrecv_kernel, 2, 1, params));
    CUDA_CHECK(cudaPeekAtLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    REQUIRE(ok[0] == true);
    REQUIRE(ok[1] == true);
}

__global__ void network_flood_kernel(
    CudaMPI::SharedState* sharedState,
    CudaMPI::ThreadPrivateState::Context threadPrivateStateContext,
    bool* ok)
{
    CudaMPI::setSharedState(sharedState);
    CudaMPI::ThreadPrivateState::Holder threadPrivateStateHolder(threadPrivateStateContext);

    const int numRanks = 3;
    
    int thisRank = cg::this_grid().thread_rank();
    int nextRank = (thisRank + 1) % numRanks;
    int prevRank = (numRanks + thisRank - 1) % numRanks;
    
    const int numRepeats = 5;
    const int dataSize = 1 << (numRepeats - 1);
    
    int localData[dataSize] = {};
    for (int i = 0; i < dataSize; i++) {
        localData[i] = (thisRank + 1) * i;
    }
    
    int prevData[dataSize] = {};
    int nextData[dataSize] = {};
    
    
    CudaMPI::PendingOperation* send_next_op[numRepeats];
    CudaMPI::PendingOperation* recv_next_op[numRepeats];
    
    CudaMPI::PendingOperation* send_prev_op[numRepeats];
    CudaMPI::PendingOperation* recv_prev_op[numRepeats];
    
    for (int i = 0; i < numRepeats; i++) {
        int tag = i + 10;
        send_next_op[i] = CudaMPI::isend(nextRank, localData, sizeof(int) * (1 << i), 0, tag);
        recv_next_op[i] = CudaMPI::irecv(nextRank, nextData, sizeof(int) * (1 << i), 0, tag);
        send_prev_op[i] = CudaMPI::isend(prevRank, localData, sizeof(int) * (1 << i), 0, tag);
        recv_prev_op[i] = CudaMPI::irecv(prevRank, prevData, sizeof(int) * (1 << i), 0, tag);
    }
    
    for (int i = 0; i < numRepeats; i++) {
        CudaMPI::wait(recv_prev_op[i]);
        CudaMPI::wait(send_next_op[i]);
        CudaMPI::wait(send_prev_op[i]);
        CudaMPI::wait(recv_next_op[i]);
    }
    
    ok[thisRank] = true;
    for (int i = 0; i < dataSize; i++) {
        if (prevData[i] != (prevRank + 1) * i) ok[thisRank] = false;
        if (nextData[i] != (nextRank + 1) * i) ok[thisRank] = false;
    }
}

TEST_CASE("Network flood", "[network_flood]") {
    CudaMPI::SharedState::Context sharedStateContext = {3, 10, 10, 10, 10};
    CudaMPI::SharedState::Holder sharedStateHolder(sharedStateContext);
    CudaMPI::ThreadPrivateState::Context threadPrivateStateContext = {20};

    bool* ok;
    CUDA_CHECK(cudaMallocManaged(&ok, 3 * sizeof(bool)));
    ok[0] = false;
    ok[1] = false;
    ok[2] = false;

    CudaMPI::SharedState* sharedStatePtr = sharedStateHolder.get();
    
    void* params[] = {
        (void*)&sharedStatePtr, 
        (void*)&threadPrivateStateContext,
        (void*)&ok
    };
    
    CUDA_CHECK(cudaLaunchCooperativeKernel((void*)network_flood_kernel, 3, 1, params));
    CUDA_CHECK(cudaPeekAtLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    REQUIRE(ok[0] == true);
    REQUIRE(ok[1] == true);
    REQUIRE(ok[2] == true);
}
