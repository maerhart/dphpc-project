
struct KeyValue
{
    uint32_t key;
    uint32_t value;
}

const uint32_t capacity = 1024 * 1024; // 8 MB per block

const uint32_t max_chain = capacity / 2;

const uint32_t empty = 0xffffffff;

// 32 bit Murmur3 hash
__device__ uint32_t hash(uint32_t k)
{
    k ^= k >> 16;
    k *= 0x85ebca6b;
    k ^= k >> 13;
    k *= 0xc2b2ae35;
    k ^= k >> 16;
    return k & (capacity-1);
}

__device__ void create_hashtable(KeyValue** table) {
   *table = (KeyValue *) malloc(capacity * sizeof(KeyValue));
   if(*table != NULL)
       memset(*table, 0xff, sizeof(KeyValue) * capacity);
}

__device__ void insert(KeyValue* hashtable, KeyValue* kv, bool *success) {
    uint32_t key = kv->key;
    uint32_t value = kv->value;
    uint32_t slot = hash(key);
    uint32_t slot_start = slot;
    while (true) {
        uint32_t prev = atomicCAS(&hashtable[slot].key, empty, key);
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
    uint32_t key = kv->key;
    uint32_t slot = hash(key);

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
    uint32_t key = kv->key;
    uint32_t slot = hash(key);

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

__device__ void *dyn_malloc(size_t size, bool coalesced) {
    if(!coalesced) {
        // Ensure compatibility with coalesced case
        // | counter 4B | returned ptr... |
        void *ptr = malloc(size + sizeof(int));
        int *counter = (int*) ptr;
        *counter = 1;
        return counter+1;
    }

    if (threadIdx.x == 0) {
        *mem = malloc((size + sizeof(int*)) * blockDim.x + sizeof(int));
        // Initialize counter
        **(int**)mem = blockDim.x;
    }

    auto block = cooperative_groups::this_thread_block();

    block.sync();

    void *ptr = (char*)*mem + sizeof(int) + sizeof(int*) + threadIdx.x * (size + sizeof(int*));
    int **counter_ptr = (int**) ((char*)ptr-sizeof(int*));
    *counter_ptr = (int*)*mem;

    block.sync();

    return ptr;
}

__device__ void dyn_free(void *memptr) {
    int *counter_ptr = *((int**)memptr-1);
    int counter = atomicAdd(counter_ptr, -1);

    if (counter==0) {
        free(counter_ptr);
    }
}
