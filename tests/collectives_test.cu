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
        for (int i = 0; i < bufsize * comm_size; i++) {
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


struct ScatterTest {
    static __device__ void run(bool& ok) {
        MPI_Init(nullptr, nullptr);

        int comm_rank = -1;
        MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
        int comm_size = -1;
        MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

        int root = 2;

        int bufsize = 4;
        CudaMPI::DeviceVector<int> send_buffer(bufsize * comm_size);
        CudaMPI::DeviceVector<int> recv_buffer(bufsize);


        if (comm_rank == root) {
            for (int j = 0; j < comm_size; j++) {
                for (int i = 0; i < bufsize; i++) {
                    send_buffer[i + j * bufsize] = j;
                }
            }
        } else {
            for (int j = 0; j < comm_size; j++) {
                for (int i = 0; i < bufsize; i++) {
                    send_buffer[i + j * bufsize] = -2;
                }
            }
        }

        for (int i = 0; i < bufsize; i++) {
            recv_buffer[i] = -1;
        }

        MPI_Scatter(&send_buffer[0], bufsize, MPI_INT, &recv_buffer[0], bufsize, MPI_INT, root, MPI_COMM_WORLD);

        ok = true;
        for (int i = 0; i < bufsize; i++) {
            if (recv_buffer[i] != comm_rank) {
                printf("ERROR: process %d, recv_buffer[%d] = %d\n", comm_rank, i, recv_buffer[i]);
                ok = false;
                break;
            }
        }

        MPI_Finalize();
    }
};

TEST_CASE("MPI_Scatter test", "[scatter test]") {
    TestRunner testRunner(5);
    testRunner.run<ScatterTest>();
}


struct ScattervTest {
    static __device__ void run(bool& ok) {
        MPI_Init(nullptr, nullptr);

        int comm_rank = -1;
        MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
        int comm_size = -1;
        MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

        int root = 2;

        CudaMPI::DeviceVector<int> sizes(comm_size);
        CudaMPI::DeviceVector<int> displs(comm_size);

        for (int i = 0; i < comm_size; i++) {
            sizes[i] = i + 1;
            if (i == 0) {
                displs[i] = 0;
            } else {
                displs[i] = displs[i - 1] + sizes[i - 1];
            }
        }

        int total_size = displs[comm_size - 1] + sizes[comm_size - 1];

        CudaMPI::DeviceVector<int> send_buffer(total_size);
        CudaMPI::DeviceVector<int> recv_buffer(sizes[comm_rank]);

        for (int j = 0; j < comm_size; j++) {
            for (int i = 0; i < sizes[j]; i++) {
                send_buffer[displs[j] + i] = (comm_rank == root) ? j : -2;
            }
        }

        for (int i = 0; i < sizes[comm_rank]; i++) {
            recv_buffer[i] = -1;
        }

        MPI_Scatterv(&send_buffer[0], &sizes[0], &displs[0], MPI_INT, &recv_buffer[0], sizes[comm_rank], MPI_INT, root, MPI_COMM_WORLD);

        if (comm_rank == root) {
            printf("Send buffer: ");
            for (int i = 0; i < total_size; i++) {
                printf("%d ", send_buffer[i]);
            }
            printf("\n");

            printf("Sizes: ");
            for (int i = 0; i < comm_size; i++) {
                printf("%d ", sizes[i]);
            }
            printf("\n");

            printf("displs: ");
            for (int i = 0; i < comm_size; i++) {
                printf("%d ", displs[i]);
            }
            printf("\n");
        }

        MPI_Barrier(MPI_COMM_WORLD);

        for (int j = 0; j < comm_size; j++) {
            if (j == comm_rank) {
                printf("Recv buffer %d: ", comm_rank);
                for (int i = 0; i < sizes[comm_rank]; i++) {
                    printf("%d ", recv_buffer[i]);
                }
                printf("\n");
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }

        ok = true;
        for (int i = 0; i < sizes[comm_rank]; i++) {
            if (recv_buffer[i] != comm_rank) {
                printf("ERROR: process %d, recv_buffer[%d] = %d\n", comm_rank, i, recv_buffer[i]);
                ok = false;
                break;
            }
        }

        MPI_Finalize();
    }
};

TEST_CASE("MPI_Scatterv test", "[scatterv test]") {
    TestRunner testRunner(5);
    testRunner.run<ScattervTest>();
}

