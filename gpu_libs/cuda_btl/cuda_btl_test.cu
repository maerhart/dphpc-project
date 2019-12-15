#include "cuda_send_recv.cuh"

#include <cuda.h>
#include <cooperative_groups.h>

#include <vector>
#include <memory>

#define VOLATILE(x) (*((volatile decltype(x)*)&x))



namespace cg = cooperative_groups;

namespace CudaMPI {


enum { ANY_SRC = -1 };
enum { ANY_TAG = -1 };

struct PendingOperation {
    enum class Type { SEND, RECV };
    
    // one of two state transitions are possible
    // STARTED -> POSTED -> SYNCED -> COMPLETED
    // STARTED -> MATCHED -> ALLOCATED -> SYNCED -> COMPLETED
    enum class State {
        STARTED,
        POSTED,
        MATCHED,
        ALLOCATED,
        SYNCED,
        COMPLETED
    };
    
    Type type = Type::SEND;
    State state = State::STARTED;
    volatile MemoryFragment* fragment = nullptr;
    int otherThread = 0;
    PendingOperation* foreignPendingOperation = nullptr;
    void* data = nullptr;
    int count = 0;
    int comm = 0;
    int tag = 0;
    bool canBeFreed = false;
    bool unused = true;

    __device__ void free() { unused = true; }
};

__device__ void progress();

__device__ void progressSend(PendingOperation& send);
__device__ void progressRecv(PendingOperation& recv);

__device__ void progressStartedRecv(PendingOperation& recv);
__device__ void progressPostedRecv(PendingOperation& recv);
__device__ void progressMatchedRecv(PendingOperation& recv);
__device__ void progressAllocatedRecv(PendingOperation& recv);
__device__ void progressSyncedRecv(PendingOperation& recv);
__device__ void progressCompletedRecv(PendingOperation& recv);

__device__ void progressStartedSend(PendingOperation& send);
__device__ void progressPostedSend(PendingOperation& send);
__device__ void progressMatchedSend(PendingOperation& send);
__device__ void progressAllocatedSend(PendingOperation& send);
__device__ void progressSyncedSend(PendingOperation& send);
__device__ void progressCompletedSend(PendingOperation& send);

struct ThreadPrivateState {
    __device__ explicit ThreadPrivateState(int pendingBufferSize)
        : pendingOperations(pendingBufferSize)
    {
    }

    __device__ PendingOperation* allocatePendingOperation() {
        for (int i = 0; i < pendingOperations.size(); i++) {
            PendingOperation& op = pendingOperations[i];
            if (op.unused) {
                op.unused = false;
                return &op;
            }
        }
        return nullptr;
    }

    __device__ Vector<PendingOperation>& getPendingOperations() { return pendingOperations; }

private:

    Vector<PendingOperation> pendingOperations;
};

struct MessageDescriptor {
    PendingOperation* privatePointer;
    int src;
    int comm;
    int tag;

    __host__ __device__ volatile MessageDescriptor& operator=(const MessageDescriptor& other) volatile {
        privatePointer = other.privatePointer;
        src = other.src;
        comm = other.comm;
        tag = other.tag;
        return *this;
    }
};

struct IncomingFragment {
    volatile MemoryFragment* fragment;
    PendingOperation* privatePointer;
    
    __host__ __device__ volatile IncomingFragment& operator=(const IncomingFragment& other) volatile {
        fragment = other.fragment;
        privatePointer = other.privatePointer;
        return *this;
    }
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
private:
    SharedState(
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

    static void destroy(SharedState* sharedState) {
        sharedState->~SharedState();
        CUDA_CHECK(cudaFree(sharedState));
    }

public:
    using UniquePtr = std::unique_ptr<SharedState, decltype(&destroy)>;

    template <typename... Args>
    static UniquePtr allocate(Args... args) {
        SharedState* ret = nullptr;
        CUDA_CHECK(cudaMallocManaged(&ret, sizeof(SharedState)));
        new (ret) SharedState(args...);
        assert(ret);
        return UniquePtr(ret, &destroy);
    }

    ManagedVector<SharedThreadState> sharedThreadState;
    SharedFragmentBuffer sharedFragmentBuffer;
};

__device__ SharedState* gSharedState = nullptr;

__device__ SharedState& sharedState() {
    assert(gSharedState != nullptr);
    return *gSharedState;
};


// this pointer should be initialized before executing any other functions
// size of this array should be equal to the number of launched threads
// on this device
__device__ ThreadPrivateState* gThreadLocalState = nullptr;

__device__ ThreadPrivateState& threadPrivateState() {
    assert(gThreadLocalState != nullptr);
    int gridIdx = cg::this_grid().thread_rank();
    return gThreadLocalState[gridIdx];
}

template <typename... Args>
__device__ void initializeThreadPrivateState(Args... args) {
    LOG("initializeThreadPrivateState()");
    if (0 == cg::this_grid().thread_rank()) {
        *((volatile ThreadPrivateState**)&gThreadLocalState) = (ThreadPrivateState*)malloc(cg::this_grid().size() * sizeof(ThreadPrivateState));
    }
    cg::this_grid().sync();
    new (&threadPrivateState()) ThreadPrivateState(args...);
}

__device__ void destroyThreadPrivateState() {
    LOG("destroyThreadPrivateState()");
    threadPrivateState().~ThreadPrivateState();
    cg::this_grid().sync();
    if (0 == cg::this_grid().thread_rank()) {
        free(gThreadLocalState);
    }
}



__device__ PendingOperation* isend(int dst, const void* data, int count, int comm, int tag) {
    LOG("isend");
    PendingOperation* po = threadPrivateState().allocatePendingOperation();
    while (!po) {
        po = threadPrivateState().allocatePendingOperation();
        LOG("WARNING: Pending operations limit is reached, isend can be blocked\n");
        progress();
    }

    po->type = PendingOperation::Type::SEND;
    po->state = PendingOperation::State::STARTED;
    po->fragment = nullptr;
    po->otherThread = dst;
    po->data = (void*) data;
    po->count = count;
    po->comm = comm;
    po->tag = tag;
    po->unused = false;

    progress();

    return po;
}

__device__ PendingOperation* irecv(int src, void* data, int count, int comm, int tag) {
    LOG("irecv");

    PendingOperation* po = threadPrivateState().allocatePendingOperation();
    while (!po) {
        po = threadPrivateState().allocatePendingOperation();
        printf("WARNING: Pending operations limit is reached, irecv can be blocked\n");
        progress();
    }

    po->type = PendingOperation::Type::RECV;
    po->state = PendingOperation::State::STARTED;
    po->fragment = nullptr;
    po->otherThread = src;
    po->data = data;
    po->count = count;
    po->comm = comm;
    po->tag = tag;
    po->unused = false;

    progress();

    return po;
}

__device__ void progressCompletedRecv(PendingOperation& recv) {
    LOG("progressCompletedRecv");

    if (recv.canBeFreed) {
        LOG("freeing local recv operation");
        recv.free();
    }
}

__device__ void progressCompletedSend(PendingOperation& send) {
    LOG("progressCompletedSend");

    LOG("unlocking memory fragment");
    send.fragment->memoryLock.unlock();
    if (send.canBeFreed) {
        LOG("freeing local send operation");
        send.free();
    }
}

__device__ void progressAllocatedSend(PendingOperation& send) {
    LOG("progressAllocatedSend()");

    volatile SharedThreadState* threadState = sharedState().sharedThreadState.get(send.otherThread);
    LOG("trying to lock incoming fragments of other thread %d", send.otherThread);
    if (!threadState->fragLock.tryLock()) {
        LOG("fragment lock failed");
        return;
    }
    LOG("fragment lock succeed");

    IncomingFragment fr;
    assert(send.fragment); // fragment should be allocated
    fr.fragment = send.fragment;
    fr.privatePointer = send.foreignPendingOperation;

    LOG("put fragment %p into list of incoming fragments", fr.fragment);
    threadState->incomingFragments.push(fr);

    LOG("unlocking list of incoming fragments");
    threadState->fragLock.unlock();

    if (send.count == 0) {
        LOG("All buffer data is already inside fragment, change state to COMPLETED");
        send.state = PendingOperation::State::COMPLETED;
        progressCompletedSend(send);
    } else {
        LOG("Change state to SYNCED (fragment now on the other thread)");
        send.state = PendingOperation::State::SYNCED;
        progressSyncedSend(send);
    }
}

__device__ void progressMatchedSend(PendingOperation& send) {
    LOG("progressMatchedSend()");

    LOG("Trying to allocate memory fragment");
    SharedFragmentBuffer& fb = sharedState().sharedFragmentBuffer;
    volatile MemoryFragment* memoryFragment = fb.tryLockFreeFragment();
    if (!memoryFragment) {
        LOG("Memory fragment allocation is failed");
        return;
    }
    LOG("Memory fragment allocation is succeed");

    int copySize = 0;
    void* srcPtr = nullptr;
    LOG("Compare fragment buffer size %d and data size %d", memoryFragment->data.size(), send.count);
    if (memoryFragment->data.size() >= send.count) {
        LOG("Fragment buffer size greater or equal to data size");
        copySize = memoryFragment->data.size();
        srcPtr = send.data;
        send.data = nullptr;
        send.count = 0;
        send.state = PendingOperation::State::COMPLETED;
    } else {
        LOG("Fragment buffer size less than data size");
        copySize = send.count;
        srcPtr = send.data;
        send.data = (void*)(((char*)send.data) + copySize);
        send.count -= copySize;
        LOG("Change state to allocated");
        send.state = PendingOperation::State::ALLOCATED;
    }
    LOG("Copying data from local memory into memory fragment");
    memcpy_volatile(memoryFragment->data.get(0), srcPtr, copySize);

    LOG("Transfer ownership of memory fragment to thread %d", send.otherThread);
    memoryFragment->ownerProcess = send.otherThread;

    LOG("Memory fragment of local pending operation is set to %p", memoryFragment);
    send.fragment = memoryFragment;

    if (send.state == PendingOperation::State::ALLOCATED) {
        progressAllocatedSend(send);
    } else if (send.state == PendingOperation::State::COMPLETED) {
        progressCompletedSend(send);
    }
}

__device__ void progressStartedSend(PendingOperation& send) {
    LOG("progressStartedSend()");
    volatile SharedThreadState* otherThreadState = sharedState().sharedThreadState.get(send.otherThread);

    int src = cg::this_grid().thread_rank();

    LOG("Trying to lock state of other process");
    if (!otherThreadState->recvLock.tryLock()) {
        LOG("Failed to lock state of other process");
        return;
    }
    LOG("State of other process is locked");

    volatile CircularQueue<MessageDescriptor>& uq = otherThreadState->unexpectedRecv;
    volatile CircularQueue<MessageDescriptor>& rq = otherThreadState->expectedRecv;

    volatile MessageDescriptor* matchedRecv = nullptr;

    LOG("Trying to find matching send in the list of expected receives of other process");
    for (volatile MessageDescriptor* md = rq.head(); md != nullptr; md = rq.next(md)) {
        if (md->src != ANY_SRC && md->src != src) continue;
        if (md->comm != send.comm) continue;
        if (md->tag != ANY_TAG && md->tag != send.tag) continue;
        // if we are here then "md" matches "send"
        matchedRecv = md;
        LOG("Matching receive is found!");
        break;
    }

    if (matchedRecv) {
        LOG("Remove receive from the list of expected receives of other process");
        send.foreignPendingOperation = matchedRecv->privatePointer;
        rq.pop(matchedRecv);
        LOG("Change state to MATCHED");
        send.state = PendingOperation::State::MATCHED;
    } else {
        LOG("Matching receive is not found, post send in unexpected receives of other process");
        
        MessageDescriptor md;
        md.comm = send.comm;
        md.src = src;
        md.tag = send.tag;
        uq.push(md);
        LOG("Change state to POSTED");
        send.state = PendingOperation::State::POSTED;
    }

    LOG("Unlock state of other process");
    otherThreadState->recvLock.unlock();

    if (send.state == PendingOperation::State::MATCHED) {
        progressMatchedSend(send);
    } else if (send.state == PendingOperation::State::POSTED) {
        progressPostedSend(send);
    }
}


__device__ void progressPostedSend(PendingOperation& send) {
    LOG("progressPostedSend()");

    if (send.fragment != nullptr) {
        LOG("Fragment is allocated by other thread, change state to SYNCED");
        send.state = PendingOperation::State::SYNCED;
        progressSyncedSend(send);
    } else {
        LOG("Fragment is not allocated by other thread, skip it");
    }
}

__device__ void progressSyncedSend(PendingOperation& send) {
    LOG("progressSyncedSend()");

    LOG("check the owner of shared fragment buffer");
    if (send.fragment->ownerProcess == send.otherThread) {
        LOG("buffer is owned by other thread, skip it");
        return;
    }
    LOG("buffer is owned by me, continue operation");

    int copySize = 0;
    void* srcPtr = nullptr;
    if (send.fragment->data.size() < send.count) {
        LOG("copy next chunk, it is not the last one");
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
        LOG("copy last chunk, change state to COMPLETED");
        send.state = PendingOperation::State::COMPLETED;
    }
    LOG("copy chunk from local buffer to destionation buffer");
    memcpy_volatile(send.fragment->data.get(0), srcPtr, copySize);

    LOG("transfer ownership of shared fragment to other thread");
    send.fragment->ownerProcess = send.otherThread;

    if (send.state == PendingOperation::State::COMPLETED) {
        progressCompletedSend(send);
    }
}

__device__ void progressSend(PendingOperation& send) {
    LOG("progressSend()");

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
        case PendingOperation::State::COMPLETED:
            progressCompletedSend(send);
            break;
    }
}


__device__ void progressStartedRecv(PendingOperation& recv) {
    LOG("progressStartedRecv()");

    int dst = cg::this_grid().thread_rank();

    volatile SharedThreadState* currentThreadState = sharedState().sharedThreadState.get(dst);

    LOG("Trying to take lock for shared thread state of current thread");
    if (!currentThreadState->recvLock.tryLock()) {
        LOG("Failed to take lock");
        return;
    }
    LOG("Lock is taken successfully");

    volatile CircularQueue<MessageDescriptor>& uq = currentThreadState->unexpectedRecv;
    volatile CircularQueue<MessageDescriptor>& rq = currentThreadState->expectedRecv;

    volatile MessageDescriptor* matchedSend = nullptr;

    LOG("Trying to find message in the list of unexpected messages");
    for (volatile MessageDescriptor* md = uq.head(); md != nullptr; md = uq.next(md)) {
        if (md->comm != recv.comm) continue;
        if (md->tag != recv.tag) continue;
        // if we are here then "md" matches "recv"
        LOG("Message is found in unexpected list");
        matchedSend = md;
        break;
    }

    if (matchedSend) {
        LOG("Save pointer to `send` operation of other process");
        recv.foreignPendingOperation = matchedSend->privatePointer;
        LOG("Remove message from list of unexpected messages");
        uq.pop(matchedSend);
        
        LOG("Change state to MATCHED");
        recv.state = PendingOperation::State::MATCHED;
    } else {
        LOG("Add message to the list of expected receives of current threads");
        MessageDescriptor md;
        md.comm = recv.comm;
        md.src = recv.otherThread;
        md.tag = recv.tag;
        rq.push(md);
        
        LOG("Change state to POSTED");
        recv.state = PendingOperation::State::POSTED;
    }

    LOG("Unlock shared state of current thread");
    currentThreadState->recvLock.unlock();

    if (recv.state == PendingOperation::State::MATCHED) {
        progressMatchedRecv(recv);
    } else if (recv.state == PendingOperation::State::POSTED) {
        progressPostedRecv(recv);
    }
}


__device__ void progressPostedRecv(PendingOperation& recv) {
    LOG("progressPostedRecv()");

    if (recv.fragment != nullptr) {
        LOG("Fragment is allocated by other thread, change state to SYNCED");
        recv.state = PendingOperation::State::SYNCED;
        progressSyncedRecv(recv);
    } else {
        LOG("Fragment is not allocated by other thread, skip it");
    }
}

__device__ void progressMatchedRecv(PendingOperation& recv) {
    LOG("progressMatchedRecv()");

    LOG("Trying lock free memory fragment");
    SharedFragmentBuffer& fb = sharedState().sharedFragmentBuffer;
    volatile MemoryFragment* memoryFragment = fb.tryLockFreeFragment();
    if (!memoryFragment) {
        LOG("Failed to lock memory fragment");
        return;
    }
    LOG("Memory fragment is locked");
    
    LOG("Transfer ownership of fragment to other thread");
    memoryFragment->ownerProcess = recv.otherThread;
    
    recv.fragment = memoryFragment;
    
    LOG("Change state to ALLOCATED");
    recv.state = PendingOperation::State::ALLOCATED;
    
    progressAllocatedRecv(recv);
}

__device__ void progressAllocatedRecv(PendingOperation& recv) {
    LOG("progressAllocatedRecv()");

    LOG("Trying to lock list of incoming fragments of thread %d", recv.otherThread);
    volatile SharedThreadState* threadState = sharedState().sharedThreadState.get(recv.otherThread);
    if (!threadState->fragLock.tryLock()) {
        LOG("Failed to lock");
        return;
    }
    LOG("Locked successfully");
    
    
    IncomingFragment fr;
    fr.fragment = recv.fragment;
    fr.privatePointer = recv.foreignPendingOperation;
    
    assert(fr.fragment);
    
    LOG("Put new fragment into list of incoming fragments");
    threadState->incomingFragments.push(fr);
    
    LOG("Unlock list of incoming fragments of other thread %d", recv.otherThread);
    threadState->fragLock.unlock();
    
    LOG("Change state to SYNCED");
    recv.state = PendingOperation::State::SYNCED;
    
    progressSyncedRecv(recv);
}

__device__ void progressSyncedRecv(PendingOperation& recv) {
    LOG("progressSyncedRecv()");

    LOG("Check that current thread owns fragment");
    if (recv.fragment->ownerProcess == recv.otherThread) {
        LOG("Fragment is used by other process, skip it");
        return;
    }
    LOG("Fragment is owned by current thread");
    
    int copySize = 0;
    void* dstPtr = nullptr;
    if (recv.fragment->data.size() < recv.count) {
        LOG("Prepare copy of next chank");
        // a lot of chunks left
        copySize = recv.fragment->data.size();
        dstPtr = recv.data;
        recv.data = (void*)((char*)recv.data + copySize);
        recv.count -= copySize;
    } else {
        LOG("Prepare copy of last chank");
        // last chunk
        copySize = recv.count;
        dstPtr = recv.data;
        recv.data = nullptr;
        recv.count = 0;
        LOG("Change state to COMPLETED");
        recv.state = PendingOperation::State::COMPLETED;
    }
    LOG("Copy data from fragment buffer into local memory");
    memcpy_volatile(dstPtr, recv.fragment->data.get(0), copySize);
    
    LOG("Transfer fragment ownership to other thread");
    recv.fragment->ownerProcess = recv.otherThread;
    
    if (recv.state == PendingOperation::State::COMPLETED) {
        progressCompletedRecv(recv);
    }
}

__device__ void progressRecv(PendingOperation& recv) {
    LOG("progressRecv()");

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
        case PendingOperation::State::COMPLETED:
            progressCompletedRecv(recv);
            break;
    }
}

__device__ void receiveFragmentPointers() {
    LOG("receiveFragmentPointers()");

    int curThread = cg::this_grid().thread_rank();
    SharedState& ss = sharedState();
    volatile SharedThreadState* sts = ss.sharedThreadState.get(curThread);

    LOG("Trying to lock list of incoming fragment of current thread");
    if (!sts->fragLock.tryLock()) {
        LOG("Failed to lock");
        return;
    }
    LOG("Locked successfully");
    
    LOG("Looping over incoming fragments");
    while (!sts->incomingFragments.empty()) {
        volatile IncomingFragment* inFrag = sts->incomingFragments.head();
        assert(inFrag);

        volatile MemoryFragment* frag = inFrag->fragment;
        
        PendingOperation* pop = inFrag->privatePointer;
        LOG("Extract pointer to private pending operation %p", pop);
        assert(pop);
        
        assert(pop->fragment);
        LOG("Assign incoming fragment %p to the private pending operation %p", frag, pop);
        pop->fragment = frag;
        
        LOG("Remove fragment from the list of incoming fragments");
        sts->incomingFragments.pop(inFrag);
    }
    
    LOG("Unlock list of incoming fragments of current thread");
    sts->fragLock.unlock();
}

__device__ void progress() {
    LOG("progress()");

    receiveFragmentPointers();
    
    Vector<PendingOperation>& pops = threadPrivateState().getPendingOperations();
    for (int i = 0; i < pops.size(); i++) {
        PendingOperation& pop = pops[i];
        if (!pop.unused) {
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
}

__device__ bool test(PendingOperation* op) {
    LOG("test()");
    assert(op->canBeFreed == false);
    progress();
    if (op->state == PendingOperation::State::COMPLETED) {
        op->canBeFreed = true;
        switch (op->type) {
            case PendingOperation::Type::SEND:
                progressCompletedSend(*op);
                break;
            case PendingOperation::Type::RECV:
                progressCompletedRecv(*op);
                break;
        }
        return true;
    }
    return false;
}

__device__ void wait(PendingOperation* op) {
    LOG("wait()");
    assert(op->canBeFreed == false);
    while (!test(op)) {}
}

} // namespace

__global__ void mykernel(CudaMPI::SharedState* sharedState) {

    if (cg::this_grid().thread_rank() == 0) {
        CudaMPI::gSharedState = sharedState;
    }
    cg::this_grid().sync();

    CudaMPI::initializeThreadPrivateState(20);

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
    cg::this_grid().sync();

    CudaMPI::destroyThreadPrivateState();
}

int main() {
    CudaMPI::SharedState::UniquePtr sharedState = CudaMPI::SharedState::allocate(2, 10, 10, 10, 10);
    mykernel<<<1,2>>>(sharedState.get());
    CUDA_CHECK(cudaPeekAtLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}
