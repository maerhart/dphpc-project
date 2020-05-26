#include "test_runner.cuh"

#include "mpi.h.cuh"
#include "device_vector.cuh"

struct ReduceTest {
    static __device__ void run(bool& ok) {
        MPI_Init(nullptr, nullptr);

        int comm_rank = -1;
        MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
        int comm_size = -1;
        MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

        constexpr int bufsize = 14;
        int16_t buffer[bufsize] = {};
        int16_t recv_buffer[bufsize] = {};

        for (int i = 0; i < bufsize; i++) {
            buffer[i] = (comm_rank + 1) * (i + 1);
        }

        MPI_Reduce(buffer, recv_buffer, bufsize, MPI_INT16_T, MPI_PROD, 0, MPI_COMM_WORLD);

        int16_t expected[bufsize] = {};
        for (int i = 0; i < bufsize; i++) {
            expected[i] = 1;
        }
        for (int r = 0; r < comm_size; r++) {
            for (int i = 0; i < bufsize; i++) {
                expected[i] *= (r + 1) * (i + 1);
            }
        }

        ok = true;

        if (comm_rank == 0) {
            for (int i = 0; i < bufsize; i++) {
                if (recv_buffer[i] != expected[i]) {
                    ok = false;
                    break;
                }
            }
        }

        MPI_Finalize();
    }
    
};

TEST_CASE("MPI_Reduce test", "[reduce test]") {
    TestRunner testRunner(5);
    testRunner.run<ReduceTest>();
}


struct AllReduceTest {
    static __device__ void run(bool& ok) {
        MPI_Init(nullptr, nullptr);

        int comm_rank = -1;
        MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
        int comm_size = -1;
        MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

        constexpr int bufsize = 17;
        double buffer[bufsize] = {};
        double recv_buffer[bufsize] = {};

        for (int i = 0; i < bufsize; i++) {
            buffer[i] = (comm_rank + 1) * (i + 1);
        }

        MPI_Allreduce(buffer, recv_buffer, bufsize, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        double expected[bufsize] = {};
        for (int r = 0; r < comm_size; r++) {
            for (int i = 0; i < bufsize; i++) {
                expected[i] += (r + 1) * (i + 1);
            }
        }

        ok = true;
        for (int i = 0; i < bufsize; i++) {
            if (abs(recv_buffer[i] - expected[i]) > 1e-5) {
                ok = false;
                break;
            }
        }

        MPI_Finalize();
    }
};

TEST_CASE("MPI_Allreduce test", "[all reduce test]") {
    TestRunner testRunner(7);
    testRunner.run<AllReduceTest>();
}

struct AllGatherTest {
    static __device__ void run(bool& ok) {
        MPI_Init(nullptr, nullptr);

        int comm_rank = -1;
        MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
        int comm_size = -1;
        MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

        int bufsize = 7;
        CudaMPI::DeviceVector<int> buffer(bufsize);
        CudaMPI::DeviceVector<int> recv_buffer(bufsize * comm_size);

        for (int i = 0; i < bufsize; i++) {
            buffer[i] = (comm_rank + 1) * (i + 1);
        }

        MPI_Allgather(&buffer[0], bufsize, MPI_INT, &recv_buffer[0], bufsize, MPI_INT, MPI_COMM_WORLD);

        CudaMPI::DeviceVector<int> expected(bufsize * comm_size);
        for (int r = 0; r < comm_size; r++) {
            for (int i = 0; i < bufsize; i++) {
                expected[r * bufsize + i] += (r + 1) * (i + 1);
            }
        }

        ok = true;
        for (int i = 0; i < bufsize; i++) {
            if (recv_buffer[i] != expected[i]) {
                ok = false;
                break;
            }
        }

        MPI_Finalize();
    }
};

TEST_CASE("MPI_Gather test", "[all gather test]") {
    TestRunner testRunner(7);
    testRunner.run<AllGatherTest>();
}
