#include "test_runner.cuh"

#include "mpi.h.cuh"

struct CommunicatorTest {
    static __device__ void run(bool& ok) {
        ok = true;
        
        MPI_CHECK_DEVICE(MPI_Init(nullptr, nullptr));
        
        int rank = 0;
        
        MPI_CHECK_DEVICE(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
        
        ok = (rank == cg::this_grid().thread_rank());
        
        MPI_CHECK_DEVICE(MPI_Finalize());
        
    }
};

TEST_CASE("Communicator test", "[comm test]") {
    TestRunner testRunner(5);
    testRunner.run<CommunicatorTest>();
}

struct CommunicatorSendRecvTest {
    static __device__ void run(bool& ok) {
        ok = true;
        
        MPI_CHECK_DEVICE(MPI_Init(nullptr, nullptr));
        
        int rank = 0;
        
        MPI_CHECK_DEVICE(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
        
        ok = (rank == cg::this_grid().thread_rank());
        
        MPI_CHECK_DEVICE(MPI_Finalize());
        
    }
};

TEST_CASE("CommunicatorSendRecvTest", "[comm send recv test]") {
    TestRunner testRunner(5);
    testRunner.run<CommunicatorSendRecvTest>();
}
