#include "test_runner.cuh"

#include "mpi.h.cuh"

struct SampleTest {
    static __device__ void run(bool& ok) {
        MPI_Init(nullptr, nullptr);
        ok = true;
        MPI_Finalize();
    }
};

TEST_CASE("Sample test", "[sample test]") {
    TestRunner testRunner(5);
    testRunner.run<SampleTest>();
}
