#include "test_runner.cuh"

struct SampleTest {
    static __device__ void run(bool& ok) {
        ok = true;
    }
};

TEST_CASE("Sample test", "[sample test]") {
    TestRunner testRunner(5);
    testRunner.run<SampleTest>();
}
