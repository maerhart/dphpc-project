#include "cuda_mpi.cuh"

namespace cg = cooperative_groups;

__global__ void mykernel(CudaMPI::SharedState* sharedState) {

    if (cg::this_grid().thread_rank() == 0) {
        CudaMPI::setSharedState(sharedState);
    }
    cg::this_grid().sync();

    CudaMPI::ThreadPrivateState::Holder threadPrivateStateHolder(20);

    LOG("INITIALIZE");

    if (cg::this_grid().thread_rank() == 0) {
        int x = 3456;

        CudaMPI::PendingOperation* op = CudaMPI::isend(1, &x, sizeof(int), 0, 15);

        CudaMPI::wait(op);
    } else if (cg::this_grid().thread_rank() == 1) {
        int x = -1234;

        CudaMPI::PendingOperation* op = CudaMPI::irecv(0, &x, sizeof(int), 0, 15);

        CudaMPI::wait(op);

        printf("received: %d\n", x);
    }

    LOG("FINALIZE");
}

int main() {
    CudaMPI::SharedState::Context sharedStateContext = {2, 10, 10, 10, 10};
    CudaMPI::SharedState::Holder sharedStateHolder(sharedStateContext);
    mykernel<<<1,2>>>(sharedStateHolder.get());
    CUDA_CHECK(cudaPeekAtLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}
