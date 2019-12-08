#include "cuda_send_recv.cuh"

#include <cuda.h>
#include <cooperative_groups.h>

#include <vector>

#define VOLATILE(x) (*((volatile decltype(x)*)&x))

namespace CudaMPI {

enum { ANY_SRC = -1 };
enum { ANY_TAG = -1 };

struct PendingOperation {
    enum class Type { SEND, RECV };
    
    // one of two state transitions are possible
    // STARTED -> POSTED -> SYNCED
    // STARTED -> MATCHED -> ALLOCATED -> SYNCED
    enum class State {
        STARTED,
        POSTED,
        MATCHED,
        ALLOCATED,
        SYNCED
    };
    
    Type type = Type::SEND;
    State state = State::STARTED;
    MemoryFragment* fragment = nullptr;
    int otherThread = 0;
    PendingOperation* foreignPendingOperation = nullptr;
    void* data = nullptr;
    int count = 0;
    int comm = 0;
    int tag = 0;
};

struct ThreadPrivateState {
    __device__ ThreadPrivateState(int numThreads, int pendingBufferSize) {
    }
    
    Vector<PendingOperation> pendingOperations;
};

struct MessageDescriptor {
    PendingOperation* privatePointer;
    int src;
    int comm;
    int tag;
};

struct IncomingFragment {
    MemoryFragment* fragment;
    PendingOperation* privatePointer;
};

struct SharedThreadState {
    SharedThreadState(int recvListSize, int numIncomingFragments)
        : unexpectedRecv(recvListSize)
        , expectedRecv(recvListSize)
        , incomingFragments(numIncomingFragments)
    {}
    
    ManagedMemoryLock recvLock;
    CircularQueue<MessageDescriptor> unexpectedRecv;
    CircularQueue<MessageDescriptor> expectedRecv;
    
    ManagedMemoryLock fragLock;
    CircularQueue<IncomingFragment> incomingFragments;
};

struct SharedState {
    __host__ SharedState(
        int numThreads,
        int recvListSize,
        int numFragments,
        int fragmentSize,
        int numIncomingFragments
    )
        : sharedThreadState(numThreads, recvListSize, numIncomingFragments)
        , sharedFragmentBuffer(numFragments, fragmentSize)
    {
    }
    
    ManagedVector<SharedThreadState> sharedThreadState;
    SharedFragmentBuffer sharedFragmentBuffer;
};

__device__ SharedState* gSharedState;

__device__ SharedState& sharedState() {
    return *gSharedState;
};


// this pointer should be initialized before executing any other functions
// size of this array should be equal to the number of launched threads
// on this device
__device__ ThreadPrivateState* gThreadLocalState;

__device__ ThreadPrivateState& threadPrivateState() {
    int gridIdx = cg::this_grid().thread_rank();
    return gThreadLocalState[gridIdx];
}

template <typename... Args>
__device__ void initializeThreadPrivateState(Args... args) {
    if (0 == cg::this_grid().thread_rank()) {
        *((volatile ThreadPrivateState**)&gThreadLocalState) = (ThreadPrivateState*)malloc(cg::this_grid().size() * sizeof(ThreadPrivateState));
    }
    cg::this_grid().sync();
    new (&threadPrivateState()) ThreadPrivateState(args...);
}

__device__ void destroyThreadPrivateState() {
    threadPrivateState().~ThreadPrivateState();
    cg::this_grid().sync();
    if (0 == cg::this_grid().thread_rank()) {
        free(gThreadLocalState);
    }
}

__device__ void progress();

__device__ void send(int dst, const void* data, int count, int comm, int tag) {
    PendingOperation po;
    po.type = PendingOperation::Type::SEND;
    po.state = PendingOperation::State::STARTED;
    po.fragment = nullptr;
    po.otherThread = dst;
    po.data = (void*) data;
    po.count = count;
    po.comm = comm;
    po.tag = tag;
    threadPrivateState().pendingOperations.push_back(po);
    
    progress();
}

__device__ void recv(int src, void* data, int count, int comm, int tag) {
    PendingOperation po;
    po.type = PendingOperation::Type::SEND;
    po.state = PendingOperation::State::STARTED;
    po.fragment = nullptr;
    po.otherThread = src;
    po.data = data;
    po.count = count;
    po.comm = comm;
    po.tag = tag;
    threadPrivateState().pendingOperations.push_back(po);
    
    progress();
}

__device__ void progressAllocatedSend(PendingOperation& send) {
    SharedThreadState& threadState = sharedState().sharedThreadState[send.otherThread];
    if (!threadState.fragLock.tryLock()) return;
    
    IncomingFragment fr;
    fr.fragment = send.fragment;
    fr.privatePointer = send.foreignPendingOperation;
    
    threadState.incomingFragments.push(fr);
    
    threadState.fragLock.unlock();
    
    send.state = PendingOperation::State::SYNCED;
}

__device__ void progressMatchedSend(PendingOperation& send) {
    SharedFragmentBuffer& fb = sharedState().sharedFragmentBuffer;
    MemoryFragment* memoryFragment = fb.tryLockFreeFragment();
    if (!memoryFragment) return;
    
    int copySize = 0;
    void* srcPtr = nullptr;
    if (memoryFragment->data.size() < send.count) {
        copySize = memoryFragment->data.size();
        srcPtr = send.data;
        send.data = nullptr;
        send.count = 0;
    } else {
        copySize = send.count;
        srcPtr = send.data;
        send.data = (void*)(((char*)send.data) + copySize);
        send.count -= copySize;
    }
    memcpy_volatile(&memoryFragment->data[0], srcPtr, copySize);
    
    memoryFragment->ownerProcess = send.otherThread;
    
    send.fragment = memoryFragment;
    send.state = PendingOperation::State::ALLOCATED;
    
    progressAllocatedSend(send);
}

__device__ void progressStartedSend(PendingOperation& send) {
    SharedThreadState& otherThreadState = sharedState().sharedThreadState[send.otherThread];
    
    int src = cg::this_multi_grid().thread_rank();
    
    if (!otherThreadState.recvLock.tryLock()) return;
    
    CircularQueue<MessageDescriptor>& uq = otherThreadState.unexpectedRecv;
    CircularQueue<MessageDescriptor>& rq = otherThreadState.expectedRecv;
    
    MessageDescriptor* matchedRecv = nullptr;
    
    for (MessageDescriptor* md = rq.head(); md != nullptr; md = rq.next(md)) {
        if (md->src != ANY_SRC && md->src != src) continue;
        if (md->comm != send.comm) continue;
        if (md->tag != ANY_TAG && md->tag != send.tag) continue;
        // if we are here then "md" matches "send"
        matchedRecv = md;
        break;
    }
    
    if (matchedRecv) {
        send.foreignPendingOperation = matchedRecv->privatePointer;
        rq.pop(matchedRecv);
        
        send.state = PendingOperation::State::MATCHED;
    } else {
        MessageDescriptor md;
        md.comm = send.comm;
        md.src = src;
        md.tag = send.tag;
        uq.push(md);
        
        send.state = PendingOperation::State::POSTED;
    }
    
    otherThreadState.recvLock.unlock();
    
    if (send.state == PendingOperation::State::MATCHED) {
        progressMatchedSend(send);
    }
}

__device__ void progressSyncedSend(PendingOperation& send);

__device__ void progressPostedSend(PendingOperation& send) {
    if (send.fragment != nullptr) {
        send.state = PendingOperation::State::SYNCED;
        progressSyncedSend(send);
    }
}

__device__ void progressSyncedSend(PendingOperation& send) {
    if (send.data == nullptr) {
        Vector<PendingOperation>& pops = threadPrivateState().pendingOperations;
        pops.disorderedRemove(&send - &pops[0]);
        return;
    }
    
    if (send.fragment->ownerProcess == send.otherThread) return;
    
    int copySize = 0;
    void* srcPtr = nullptr;
    if (send.fragment->data.size() < send.count) {
        // a lot of chunks left
        copySize = send.fragment->data.size();
        srcPtr = send.data;
        send.data = (void*)((char*)send.data + copySize);
        send.count -= copySize;
    } else {
        // last chunk
        copySize = send.count;
        srcPtr = send.data;
        send.data = nullptr;
        send.count = 0;
    }
    memcpy_volatile(&send.fragment->data[0], srcPtr, copySize);
    
    send.fragment->ownerProcess = send.otherThread;
    
}

__device__ void progressSend(PendingOperation& send) {
    switch (send.state) {
        case PendingOperation::State::STARTED:
            progressStartedSend(send);
            break;
        case PendingOperation::State::POSTED:
            progressPostedSend(send);
            break;
        case PendingOperation::State::MATCHED:
            progressMatchedSend(send);
            break;
        case PendingOperation::State::ALLOCATED:
            progressAllocatedSend(send);
            break;
        case PendingOperation::State::SYNCED:
            progressSyncedSend(send);
            break;
    }
}

__device__ void progressMatchedRecv(PendingOperation& recv);

__device__ void progressStartedRecv(PendingOperation& recv) {
    int dst = cg::this_multi_grid().thread_rank();
    
    SharedThreadState& currentThreadState = sharedState().sharedThreadState[dst];
    
    if (!currentThreadState.recvLock.tryLock()) return;
    
    CircularQueue<MessageDescriptor>& uq = currentThreadState.unexpectedRecv;
    CircularQueue<MessageDescriptor>& rq = currentThreadState.expectedRecv;
    
    MessageDescriptor* matchedSend = nullptr;
    
    for (MessageDescriptor* md = uq.head(); md != nullptr; md = uq.next(md)) {
        if (md->comm != recv.comm) continue;
        if (md->tag != recv.tag) continue;
        // if we are here then "md" matches "recv"
        matchedSend = md;
        break;
    }
    
    if (matchedSend) {
        recv.foreignPendingOperation = matchedSend->privatePointer;
        uq.pop(matchedSend);
        
        recv.state = PendingOperation::State::MATCHED;
    } else {
        MessageDescriptor md;
        md.comm = recv.comm;
        md.src = recv.otherThread;
        md.tag = recv.tag;
        rq.push(md);
        
        recv.state = PendingOperation::State::POSTED;
    }
    
    currentThreadState.recvLock.unlock();
    
    if (recv.state == PendingOperation::State::MATCHED) {
        progressMatchedRecv(recv);
    }
}

__device__ void progressSyncedRecv(PendingOperation& recv);

__device__ void progressPostedRecv(PendingOperation& recv) {
    if (recv.fragment != nullptr) {
        recv.state = PendingOperation::State::SYNCED;
        progressSyncedRecv(recv);
    }
}

__device__ void progressAllocatedRecv(PendingOperation& recv);


__device__ void progressMatchedRecv(PendingOperation& recv) {
    SharedFragmentBuffer& fb = sharedState().sharedFragmentBuffer;
    MemoryFragment* memoryFragment = fb.tryLockFreeFragment();
    if (!memoryFragment) return;
    
    memoryFragment->ownerProcess = recv.otherThread;
    
    recv.fragment = memoryFragment;
    recv.state = PendingOperation::State::ALLOCATED;
    
    progressAllocatedRecv(recv);
}

__device__ void progressAllocatedRecv(PendingOperation& recv) {
    SharedThreadState& threadState = sharedState().sharedThreadState[recv.otherThread];
    if (!threadState.fragLock.tryLock()) return;
    
    IncomingFragment fr;
    fr.fragment = recv.fragment;
    fr.privatePointer = recv.foreignPendingOperation;
    
    threadState.incomingFragments.push(fr);
    
    threadState.fragLock.unlock();
    
    recv.state = PendingOperation::State::SYNCED;
}

__device__ void progressSyncedRecv(PendingOperation& recv) {
    if (recv.fragment->ownerProcess == recv.otherThread) return;
    
    int copySize = 0;
    void* dstPtr = nullptr;
    if (recv.fragment->data.size() < recv.count) {
        // a lot of chunks left
        copySize = recv.fragment->data.size();
        dstPtr = recv.data;
        recv.data = (void*)((char*)recv.data + copySize);
        recv.count -= copySize;
    } else {
        // last chunk
        copySize = recv.count;
        dstPtr = recv.data;
        recv.data = nullptr;
        recv.count = 0;
    }
    memcpy_volatile(dstPtr, &recv.fragment->data[0], copySize);
    
    recv.fragment->ownerProcess = recv.otherThread;
    
    if (recv.data == nullptr) {
        recv.fragment->memoryLock.unlock();
        
        Vector<PendingOperation>& pops = threadPrivateState().pendingOperations;
        pops.disorderedRemove(&recv - &pops[0]);
        return;
    }
}

__device__ void progressRecv(PendingOperation& recv) {
    switch (recv.state) {
        case PendingOperation::State::STARTED:
            progressStartedRecv(recv);
            break;
        case PendingOperation::State::POSTED:
            progressPostedRecv(recv);
            break;
        case PendingOperation::State::MATCHED:
            progressMatchedRecv(recv);
            break;
        case PendingOperation::State::ALLOCATED:
            progressAllocatedRecv(recv);
            break;
        case PendingOperation::State::SYNCED:
            progressSyncedRecv(recv);
            break;
    }
}

__device__ void receiveFragmentPointers() {
    int curThread = cg::this_multi_grid().thread_rank();
    SharedThreadState& sts = sharedState().sharedThreadState[curThread];
    
    if (!sts.fragLock.tryLock()) return;
    
    while (0 != sts.incomingFragments.size()) {
        IncomingFragment* inFrag = sts.incomingFragments.head();
        
        MemoryFragment* frag = inFrag->fragment;
        PendingOperation* pop = inFrag->privatePointer;
        
        pop->fragment = frag;
        
        sts.incomingFragments.pop(inFrag);
    }
    
    sts.fragLock.unlock();
}

__device__ void progress() {
    receiveFragmentPointers();
    
    Vector<PendingOperation>& pops = threadPrivateState().pendingOperations;
    for (int i = 0; i < pops.size(); i++) {
        PendingOperation& pop = pops[i];
        switch (pop.type) {
            case PendingOperation::Type::SEND:
                progressSend(pop);
                break;
            case PendingOperation::Type::RECV:
                progressRecv(pop);
                break;
        }
    }
}

__global__ void mykernel(SharedState& sharedThreadState) {
    Vector<double> x(3);
    initializeThreadPrivateState(2,3);
    
//     CudaMPI::send(); // TODO
//     CudaMPI::recv(); // TODO
    
    destroyThreadPrivateState();
}

} // namespace

int main() {
    CudaMPI::SharedState sharedThreadState(1, 2, 3, 4, 5);
    mykernel<<<1,2>>>(sharedThreadState);
}
