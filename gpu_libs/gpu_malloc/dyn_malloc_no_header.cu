struct KeyValue {
    void *key;
    void *value;
}

const uint32_t capacity = 1024 * 1024; // 8 MB per block

const uint32_t max_chain = capacity / 2;

const void *empty = (0xffffffff << 32) | 0xffffffff;

// 32 bit Murmur3 hash
//__device__ uint32_t hash(uint32_t k) {
//    k ^= k >> 16;
//    k *= 0x85ebca6b;
//    k ^= k >> 13;
//    k *= 0xc2b2ae35;
//    k ^= k >> 16;
//    return k & (capacity-1);
//}

__device__ uint64_t hash(uint64_t x) {
    x ^= x >> 27;
    x *= 0x3C79AC492BA7B653UL;
    x ^= x >> 33;
    x *= 0x1C69B3F74AC4AE35UL;
    x ^= x >> 27;
    return x & (capacity-1);
}

__device__ void create_hashtable(KeyValue** table) {
   *table = (KeyValue *) malloc(capacity * sizeof(KeyValue));
   if(*table != NULL)
       memset(*table, 0xff, sizeof(KeyValue) * capacity);
}

__device__ void insert(KeyValue* hashtable, KeyValue* kv, bool *success) {
    void *key = kv->key;
    void *value = kv->value;
    uint64_t slot = hash((uint64_t) key);
    uint64_t slot_start = slot;
    while (true) {
        void *prev = atomicCAS(&hashtable[slot].key, empty, key);
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

__device__ void destroy_hashtable(KeyValue **hashtable) {
    free(*hashtable);
    *hashtable = NULL;
}

__shared__ KeyValue *table;
__shared__ static void **mem;

__device__ void *dyn_malloc(size_t size, bool coalesced) {
    //init missing
    if(!coalesced) {
        // Ensure compatibility with coalesced case
        // | counter 4B | returned ptr... |
        void *ptr = malloc(size + sizeof(uint32_t));
        uint32_t *counter = (uint32_t*) ptr;
        *counter = 1;
        KeyValue kv = {
            .key = (void *)(counter+1);
            .value = (void *)(counter+1);
        };
        bool success;
        insert(table, &kv, &success);
        if(success) {
            return counter+1;
        } else {
            return NULL;
        }
    }

    if (threadIdx.x == 0) {
        *mem = malloc(size * blockDim.x + sizeof(uint32_t));
        // Initialize counter
        **(uint32_t**)mem = blockDim.x;
    }

    auto block = cooperative_groups::this_thread_block();

    block.sync();

    void *ptr = (char*)*mem + sizeof(uint32_t) + threadIdx.x * size;
    KeyValue kv = {
        .key = (void *) ptr;
        .value = (void *) *mem;
    };

    bool success;
    insert(table, &kv, &success);

    block.sync();

    if(success) {
        return ptr;
    } else {
        return NULL;
    }
}

__device__ void dyn_free(void *memptr) {
    KeyValue kv = {
        .key = memptr;
        .value = empty;
    }

    lookup(table, &kv);
    uint32_t *counter_ptr = (uint32_t *) kv.value;
    uint32_t counter = atomicAdd(counter_ptr, -1);
    remove(table, &kv);

    if (counter == 1) {
        free(counter_ptr);
    }
}
