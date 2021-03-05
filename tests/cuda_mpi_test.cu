#include "test_runner.cuh"

#include "mpi.h.cuh"

struct SingleIntKernel {
    static __device__ void run(bool& ok)
    {
        MPI_Init(nullptr, nullptr);
        int rank = -1;
        MPI_CHECK_DEVICE(MPI_Comm_rank(MPI_COMM_WORLD, &rank));

        if (rank == 0) {
            int x = 3456;

            CudaMPI::PendingOperation* op = CudaMPI::isend(1, &x, sizeof(int), 0, 15);

            CudaMPI::wait(op);

            ok = true;
        } else if (rank == 1) {
            int x = 0;

            CudaMPI::PendingOperation* op = CudaMPI::irecv(0, &x, sizeof(int), 0, 15);

            CudaMPI::wait(op);

            ok = x == 3456;
        }
        MPI_Finalize();
    }
};

TEST_CASE("Transfer single integer", "[single_int]") {
    TestRunner testRunner(2);
    testRunner.run<SingleIntKernel>();
}

struct TransferArrayKernel {
    static __device__ void run(bool& ok) {
        MPI_Init(nullptr, nullptr);

        if (CudaMPI::sharedState().gridRank() == 0) {
            int x[16] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};

            CudaMPI::PendingOperation* op = CudaMPI::isend(1, &x, sizeof(x), 0, 15);

            CudaMPI::wait(op);
            
            ok = true;
        } else if (CudaMPI::sharedState().gridRank() == 1) {
            int x[16] = {};

            CudaMPI::PendingOperation* op = CudaMPI::irecv(0, &x, sizeof(x), 0, 15);

            CudaMPI::wait(op);

            ok = true;
            for (int i = 0; i < 16; i++) {
                if (x[i] != i) ok = false;
            }
        }
        
        MPI_Finalize();
    }
};



TEST_CASE("Transfer array", "[array]") {
    TestRunner testRunner(2);
    testRunner.mSharedStateContext = {2, 10};
    testRunner.run<TransferArrayKernel>();
}

struct SendRecvKernel {
    static __device__ void run(bool& ok) {
        MPI_Init(nullptr, nullptr);
        
        if (CudaMPI::sharedState().gridRank() == 0) {
            int x[16] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
            int y[16] = {};

            CudaMPI::PendingOperation* op[2];
            op[0] = CudaMPI::irecv(1, &y, sizeof(y), 0, 15);
            op[1] = CudaMPI::isend(1, &x, sizeof(x), 0, 15);

            CudaMPI::wait(op[0]);

            ok = true;
            for (int i = 0; i < 16; i++) {
                if (y[i] != -i) {
                    printf("thread %d, y[i] = %d\n", CudaMPI::sharedState().gridRank(), y[i]);
                    ok = false;
                }
            }

            CudaMPI::wait(op[1]);
        } else if (CudaMPI::sharedState().gridRank() == 1) {
            int x[16] = {0, -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15};
            int y[16] = {};

            CudaMPI::PendingOperation* op[2];
            op[0] = CudaMPI::irecv(0, &y, sizeof(y), 0, 15);
            op[1] = CudaMPI::isend(0, &x, sizeof(x), 0, 15);

            CudaMPI::wait(op[0]);

            ok = true;
            for (int i = 0; i < 16; i++) {
                if (y[i] != i) {
                    printf("thread %d, y[i] = %d\n", CudaMPI::sharedState().gridRank(), y[i]);
                    ok = false;
                }
            }

            CudaMPI::wait(op[1]);
        }
        
        MPI_Finalize();
    }
};

TEST_CASE("Send receive", "[send_recv]") {
    TestRunner testRunner(2);
    testRunner.mSharedStateContext = {2, 10};
    testRunner.run<SendRecvKernel>();
}

struct RepeatSendRecvKernel {
    static __device__ void run(bool& ok) {
        MPI_Init(nullptr, nullptr);

        const int numRanks = 2;
        
        int thisRank = CudaMPI::sharedState().gridRank();
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
        
        ok = true;
        for (int i = 0; i < dataSize; i++) {
            if (remoteData[i] != (otherRank + 1) * i) ok = false;
        }
        MPI_Finalize();
    }
};

TEST_CASE("Repeat send recv", "[repeat_sendrecv]") {
    TestRunner testRunner(2);
    testRunner.mSharedStateContext = {2, 10};
    testRunner.run<RepeatSendRecvKernel>();
}


struct NetworkFloodKernel {
    static __device__ void run(bool& ok) {
        MPI_Init(nullptr, nullptr);

        const int numRanks = 3;
        
        int thisRank = CudaMPI::sharedState().gridRank();
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
        
        ok = true;
        for (int i = 0; i < dataSize; i++) {
            if (prevData[i] != (prevRank + 1) * i) ok = false;
            if (nextData[i] != (nextRank + 1) * i) ok = false;
        }
        MPI_Finalize();
    }
};

TEST_CASE("Network flood", "[network_flood]") {
    TestRunner testRunner(3);
    testRunner.mSharedStateContext = {3, 10};
    testRunner.run<NetworkFloodKernel>();
}

struct AllToAllKernel {
    static __device__ void run(bool& ok) {
        MPI_Init(nullptr, nullptr);
    
        int thisRank = CudaMPI::sharedState().gridRank();
        
        const int numRepeats = 20;
        const int dataSize = 16;
        
        int numRanks;
        MPI_Comm_size(MPI_COMM_WORLD, &numRanks);
        
        int* localData = (int*) malloc(sizeof(int) * dataSize * numRepeats * numRanks);
        assert(localData);
        int* remoteData = (int*) malloc(sizeof(int) * dataSize * numRepeats * numRanks);
        assert(remoteData);
        
        for (int rank = 0; rank < numRanks; rank++) {
            for (int repeat = 0; repeat < numRepeats; repeat++) {
                for (int i = 0; i < dataSize; i++) {
                    int idx = i + repeat * dataSize + rank * dataSize * numRepeats;
                    localData[idx] = i * repeat * rank;
                    remoteData[idx] = 0;
                }
            }
        }
        
        CudaMPI::PendingOperation** send_po = (CudaMPI::PendingOperation**) malloc(numRepeats * numRanks * sizeof(CudaMPI::PendingOperation*));
        CudaMPI::PendingOperation** recv_po = (CudaMPI::PendingOperation**) malloc(numRepeats * numRanks * sizeof(CudaMPI::PendingOperation*));
        
        int tag = 15;
        int comm = 17;
        
        for (int repeat = 0; repeat < numRepeats; repeat++) {
            for (int rank = 0; rank < numRanks; rank++) {
                if (rank != thisRank) {
                    int idx = repeat * dataSize + rank * dataSize * numRepeats;
                    send_po[rank + repeat * numRanks] = CudaMPI::isend(
                        rank, localData + idx, sizeof(int) * dataSize, comm, tag);
                    recv_po[rank + repeat * numRanks] = CudaMPI::irecv(
                        rank, remoteData + idx, sizeof(int) * dataSize, comm, tag);
                }
            }
        }
        
        for (int repeat = 0; repeat < numRepeats; repeat++) {
            for (int rank = 0; rank < numRanks; rank++) {
                if (rank != thisRank) {
                    CudaMPI::wait(send_po[rank + repeat * numRanks]);
                    CudaMPI::wait(recv_po[rank + repeat * numRanks]);
                }
            }
        }
        
        ok = true;
        for (int repeat = 0; repeat < numRepeats; repeat++) {
            for (int rank = 0; rank < numRanks; rank++) {
                if (rank != thisRank) {
                    for (int i = 0; i < dataSize; i++) {
                        int idx = i + repeat * dataSize + rank * dataSize * numRepeats;
                        if (i * repeat * thisRank != remoteData[idx]) {
                            ok = false;
                            printf("thisRank = %d, i = %d, repeat = %d, rank = %d, remoteData[%d] = %d\n", thisRank, i, repeat, rank, idx, remoteData[idx]);
                        }
                    }
                }
            }
        }
        
        free(send_po);
        free(recv_po);
        free(localData);
        MPI_Finalize();
    }
};


TEST_CASE("All to all", "[all_to_all]") {
    TestRunner testRunner(10);
    testRunner.mSharedStateContext = {10, 10};
    testRunner.mThreadPrivateStateContext.pendingBufferSize = 400;
    testRunner.run<AllToAllKernel>();
}

