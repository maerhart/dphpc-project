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

    single_int_kernel<<<1,2>>>(sharedStateHolder.get(), threadPrivateStateContext, res);
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

    transfer_array_kernel<<<1,2>>>(sharedStateHolder.get(), threadPrivateStateContext, ok);
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

TEST_CASE("Send receive", "[array]") {
    CudaMPI::SharedState::Context sharedStateContext = {2, 10, 10, 10, 10};
    CudaMPI::SharedState::Holder sharedStateHolder(sharedStateContext);
    CudaMPI::ThreadPrivateState::Context threadPrivateStateContext = {20};

    bool* ok;
    CUDA_CHECK(cudaMallocManaged(&ok, 2 * sizeof(bool)));
    ok[0] = false;
    ok[1] = false;

    send_recv_kernel<<<1,2>>>(sharedStateHolder.get(), threadPrivateStateContext, ok);
    CUDA_CHECK(cudaPeekAtLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    REQUIRE(ok[0] == true);
    REQUIRE(ok[1] == true);
}
