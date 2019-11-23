#ifndef CUDA_SEND_RECV
#define CUDA_SEND_RECV

#include <cuda.h>
#include <cooperative_groups.h>

namespace THREAD_LOCALITY {
    enum Type {
        WARP,
        BLOCK,
        GRID,
        MULTI_GRID
    };
}

namespace MEMORY_LOCALITY {
    enum Type {
        GLOBAL,
        LOCAL,
        SHARED,
        CONST,
        OTHER
    };
}

namespace cg = cooperative_groups;

// check where memory is located
// from https://stackoverflow.com/questions/42519766/can-i-check-whether-an-address-is-in-shared-memory
#define DEFINE_is_xxx_memory(LOCATION) \
    __device__ bool is_ ## LOCATION ## _memory(void *ptr) {\
        int res;\
        asm("{"\
            ".reg .pred p;\n\t"\
            "isspacep." #LOCATION " p, %1;\n\t"\
            "selp.b32 %0, 1, 0, p;\n\t"\
            "}"\
            : "=r"(res): "l"(ptr));\
        return res;\
    }

DEFINE_is_xxx_memory(global) // __device__, __managed__, malloc() from kernel
DEFINE_is_xxx_memory(local) // scope-local stack variables
DEFINE_is_xxx_memory(shared) // __shared__
DEFINE_is_xxx_memory(const) // __constant__





class CudaSendRecv {
public:

    struct Location {
        int gridIndex;
        int blockIndex;
        int warpIndex;
    };
    __device__ int gridRank(int threadRankInMultiGrid) {
        return threadRankInMultiGrid / cg::this_grid().size();
    }
    __device__ int rankInGrid(int threadRankInMultiGrid) {
        return threadRankInMultiGrid % cg::this_grid().size();
    }
    __device__ int blockRank(int threadRankInMultiGrid) {
        int threadRankInGrid = rankInGrid(threadRankInMultiGrid);
        return threadRankInGrid / cg::this_thread_block().size();
    }
    __device__ int rankInBlock(int threadRankInMultiGrid) {
        int threadRankInGrid = rankInGrid(threadRankInMultiGrid);
        return threadRankInGrid % cg::this_thread_block().size();
    }
    __device__ int warpRank(int threadRankInMultiGrid) {
        int threadRankInBlock = rankInBlock(threadRankInMultiGrid);
        return threadRankInBlock / warpSize;
    }
    __device__ int rankInWarp(int threadRankInMultiGrid) {
        int threadRankInBlock = rankInBlock(threadRankInMultiGrid);
        return threadRankInBlock % warpSize;
    }

    __device__ THREAD_LOCALITY::Type getThreadLocalityType(int threadA, int threadB) {
        if (gridRank(threadA) != gridRank(threadB)) {
            return THREAD_LOCALITY::MULTI_GRID;
        } else if (blockRank(threadA) != blockRank(threadB)) {
            return THREAD_LOCALITY::GRID;
        } else if (warpRank(threadA) != warpRank(threadB)) {
            return THREAD_LOCALITY::BLOCK;
        } else {
            return THREAD_LOCALITY::WARP;
        }
    }

    __device__ MEMORY_LOCALITY::Type getMemoryLocalityType(void* ptr) {
        if (is_global_memory(ptr)) {
            return MEMORY_LOCALITY::GLOBAL;
        } else if (is_local_memory(ptr)) {
            return MEMORY_LOCALITY::LOCAL;
        } else if (is_shared_memory(ptr)) {
            return MEMORY_LOCALITY::SHARED;
        } else if (is_const_memory(ptr)) {
            return MEMORY_LOCALITY::CONST;
        } else {
            return MEMORY_LOCALITY::OTHER;
        }
    }

    __device__ void sendRecv(void* ptr, int n, int srcThread, int dstThread) {
        int thisThread = cg::this_multi_grid().thread_rank();
        if (thisThread != srcThread && thisThread != dstThread) return;

        THREAD_LOCALITY::Type threadLocality = getThreadLocalityType(srcThread, dstThread);
        MEMORY_LOCALITY::Type memoryLocality = getMemoryLocalityType(ptr);

        // check one of two: memory

        switch (threadLocality) {
            case THREAD_LOCALITY::WARP:
                sendRecvWarp(ptr, n, srcThread, dstThread);
                break;
            case THREAD_LOCALITY::BLOCK:
                sendRecvBlock(ptr, n, srcThread, dstThread);
                break;
            case THREAD_LOCALITY::GRID:
                sendRecvGrid(ptr, n, srcThread, dstThread);
                break;
            case THREAD_LOCALITY::MULTI_GRID:
                sendRecvMultiGrid(ptr, n, srcThread, dstThread);
                break;
        }
    }

    __device__ void sendRecvWarp(void* ptr, int n, int srcThread, int dstThread);
    __device__ void sendRecvBlock(void* ptr, int n, int srcThread, int dstThread);
    __device__ void sendRecvGrid(void* ptr, int n, int srcThread, int dstThread);
    __device__ void sendRecvMultiGrid(void* ptr, int n, int srcThread, int dstThread);

};

#endif
