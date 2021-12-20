#include <assert.h>
#include <iostream>
#include <stdint.h>

struct KeyValue {
    void *key;
    void *value;
};

__constant__ uint32_t capacity = 1024 * 16; // 256 KB per block, 16 times as much space as needed for max num threads 
__constant__ uint32_t max_chain = 1024 * 16 / 2; //half of capacity
__constant__ void *empty = (void *)((0xfffffffful << 32) | 0xfffffffful);
__constant__ uint32_t alignment = 16;

__device__ uint64_t hash(uint64_t x) {
    x ^= x >> 27;
    x *= 0x3C79AC492BA7B653UL;
    x ^= x >> 33;
    x *= 0x1C69B3F74AC4AE35UL;
    x ^= x >> 27;
    return x & (capacity-1);
}

__device__ void insert(KeyValue* hashtable, KeyValue* kv, bool *success) {
    void *key = kv->key;
    void *value = kv->value;
    uint64_t slot = hash((uint64_t) key);
    uint64_t slot_start = slot;
    while (true) {
        void *prev = (void *) atomicCAS((unsigned long long int *)(&hashtable[slot].key),(unsigned long long int) empty, (unsigned long long int) key);
        if (prev == empty || prev == key) {
            hashtable[slot].value = value;
            *success = true;
            return;
        }
        slot = (slot + 1) & (capacity - 1);
        if(slot == (slot_start + max_chain) % capacity) {
            *success = false;
            return;
        }
    }
}

__device__ void lookup(KeyValue *hashtable, KeyValue *kv) {
    void *key = kv->key;
    uint64_t slot = hash((uint64_t)key);

    while (true) {
        if(hashtable[slot].key == key) {
            kv->value = hashtable[slot].value;
            return;
        } else if(hashtable[slot].key == empty) {
            kv->value = empty;
            return;
        }
        slot = (slot + 1) & (capacity - 1);
    }
}

// Cannot remove the assigned slot, otherwise all keys that come after ''will get lost''
__device__ void remove(KeyValue *hashtable, KeyValue *kv) {
    void *key = kv->key;
    uint64_t slot = hash((uint64_t)key);

    while (true) {
        if(hashtable[slot].key == key) {
            hashtable[slot].value = empty;
            return;
        } else if(hashtable[slot].key == empty) {
            return;
        }
        slot = (slot + 1) & (capacity - 1);
    }
}

__shared__ KeyValue *table;
__shared__ void **mem;

__device__ void init_malloc() {
    if(!threadIdx.x) {
        table = (KeyValue *) malloc(capacity * sizeof(KeyValue));
        mem = (void **) malloc(sizeof(void *) * 32);
        if(table != NULL && mem != NULL) {
            memset(table, 0xff, sizeof(KeyValue) * capacity);
        } else {
            printf("block %i table not inited %p \n", blockIdx.x, table);
        }
    }
    __syncthreads();
}

__device__ void clean_malloc() {
    __syncthreads();
    if(!threadIdx.x) {
        free(table);
        free(mem);
        mem = NULL;
        table = NULL;
    }
    __syncthreads();
}
__device__ void* malloc_v3(size_t size) {
    int header = alignment;
    size += ((alignment - (size % alignment)) % alignment);
    if (!threadIdx.x) {
        *mem = malloc(size * blockDim.x + header);
        if (*mem == NULL) {
            printf("block %i super block of size %i failed\n", blockIdx.x, blockDim.x * (int)size);
            return NULL;
        }
        // Initialize counter
        **(int**)mem = blockDim.x;
    }

    __syncthreads();

    void *ptr = (char*)*mem + header + threadIdx.x * size;
    *((char *)ptr) = 0;
    KeyValue kv = {
        .key = (void *) ptr,
        .value = (void *) *mem
    };

    bool success;
    insert(table, &kv, &success);

    __syncthreads();

    if(success) {
        return ptr;
    } else {
        return NULL;
    }
}

__device__ void free_v3(void *memptr) {
    KeyValue kv = {
        .key = memptr,
        .value = empty
    };

    lookup(table, &kv);
    if(kv.value == empty) {
        return false;
    }
    int *counter_ptr = (int *) kv.value;
    int counter = atomicSub(counter_ptr, 1);
    remove(table, &kv);

    if (counter == 1) {
        free(counter_ptr);
    }
    return true;
}

__device__ void* malloc_v6(size_t size) {
    int header = alignment;
    size += ((alignment - (size % alignment)) % alignment);
    uint32_t warpno = threadIdx.x / 32;
    if (threadIdx.x % 32 == 0) {
        int allocates_for = blockDim.x - threadIdx.x;
        allocates_for = (allocates_for > 32) ? 32 : allocates_for;
        mem[warpno] = malloc(size * allocates_for + header);
        if (mem[warpno] == NULL) {
            printf("block %i super block of size %i failed\n", blockIdx.x, blockDim.x * (int)size);
            return NULL;
        }
        // Initialize counter
        *(int*)(mem[warpno]) = blockDim.x;
    }

    __syncwarp();

    void *ptr = (char*)(mem[warpno]) + header + (threadIdx.x % 32) * size;
    *((char *)ptr) = 0;
    KeyValue kv = {
        .key = (void *) ptr,
        .value = (void *) (mem[warpno])
    };

    bool success;
    insert(table, &kv, &success);

    __syncwarp();

    if(success) {
        return ptr;
    } else {
        return NULL;
    }
}

__device__ bool free_v6(void *memptr) {
    KeyValue kv = {
        .key = memptr,
        .value = empty
    };

    lookup(table, &kv);
    if(kv.value == empty) {
        return false;
    }
    int *counter_ptr = (int *) kv.value;
    int counter = atomicSub(counter_ptr, 1);
    remove(table, &kv);

    if (counter == 1) {
        free(counter_ptr);
    }
    return true;
}
