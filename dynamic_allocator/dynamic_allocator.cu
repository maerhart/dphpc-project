#include <assert.h>
#include <iostream>
#include <stdint.h>
#include "cooperative_groups.h"

// baseline using std malloc/free
__device__ void* malloc_baseline(size_t size) {
    void* ptr = malloc(size);
    #ifndef NDEBUG
    if (!ptr) {
        printf("GPUMPI: malloc failed to allocate %llu bytes on device\n", (long long unsigned)size);
    }
    #endif
    return ptr;
}

__device__ void free_baseline(void *memptr) {
    free(memptr);
}

// v1: allocate same sizes for future blocks

struct s_header {
    int counter;
};


// memory layout of a superblock
// s_header, blocksize x [pointer to s_header, data]
__device__ void* malloc_v1(size_t size) {
	__shared__ void* superblock;
	if (threadIdx.x == 0) {
		// allocate new superblock
		int size_superblock = sizeof(s_header) + blockDim.x * (sizeof(s_header*) + size);
		superblock = malloc(size_superblock);
		if (superblock == NULL) {
			printf("V1: failed to allocate %llu bytes on device\n", (long long unsigned)(size_superblock));
			return NULL;
		}
		// initialize header	
		struct s_header* header;
		header = (s_header*)superblock;
		header->counter = blockDim.x;
	}
	__syncthreads();

	if (superblock == NULL) return NULL;

	// ptr to individual memory offset
    s_header* ptr = (s_header*)((char*)superblock + sizeof(s_header) + threadIdx.x * (size + sizeof(s_header*)));
	// set pointer to superblock header
	*ptr = *(s_header*)superblock;
	// return the pointer to the data section
	return (void*)(ptr + 1);
}

__device__ void free_v1(void* memptr) {
	// decrease counter
	s_header* header = (s_header*)memptr - 1;    
	int count = atomicSub(&(header->counter), 1);
	// last thread frees superblock
	if (count == 1) free(header);
}



// v2: no spaceing between data, use hashmap to know mapping from address to superblock
#define SIZE_HASH_MAP 1000
__shared__ s_header* hashmap[SIZE_HASH_MAP];


// source: https://stackoverflow.com/questions/664014/what-integer-hash-function-are-good-that-accepts-an-integer-hash-key
__device__ uint64_t hash(uint64_t x) {
    x = (x ^ (x >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)) * UINT64_C(0x94d049bb133111eb);
    x = x ^ (x >> 31);
    return x;
}

__device__ void* malloc_v2(size_t size) {
    __shared__ void* superblock;
    if (threadIdx.x == 0) {
        // allocate new superblock
        int size_superblock = sizeof(s_header) + blockDim.x * size;
        superblock = malloc(size_superblock);
        if (superblock == NULL) {
            printf("V2: failed to allocate %llu bytes on device\n", (long long unsigned)(size_superblock));
            return NULL;
        }
        // initialize header
        struct s_header* header;
        header = (s_header*)superblock;
        header->counter = blockDim.x;
    }
    __syncthreads();

    if (superblock == NULL) return NULL;

    // ptr to individual memory offset
    s_header* ptr = (s_header*)((char*)superblock + sizeof(s_header) + threadIdx.x * size);
    // insert into hashmap
	int h = hash((uintptr_t)ptr) % SIZE_HASH_MAP;
	// todo: proper collision handling
	if (hashmap[h] != 0) {
		printf("V2: hash collision on %i\n", h);
	}
	hashmap[h] = (s_header*)superblock;
	// return the pointer to the data section
    return (void*)ptr;
}

__device__ void free_v2(void* memptr) {
    // get ptr and delete hashmap entry
	int h = hash((uintptr_t)memptr) % SIZE_HASH_MAP;
    s_header* header = hashmap[h];
	hashmap[h] = 0;
	// decrease counter
    int count = atomicSub(&(header->counter), 1);

    // last thread frees superblock
    if (count == 0) free(header);
}

struct KeyValue {
    void *key;
    void *value;
};

__constant__ uint32_t capacity_v3 = 1024 * 1024; // 8 MB per block
__constant__ uint32_t max_chain_v3 = 1024 * 1024 / 2; //half of capacity
__constant__ void *empty = (void *)((0xfffffffful << 32) | 0xfffffffful);

// 32 bit Murmur3 hash
//__device__ uint32_t hash(uint32_t k) {
//    k ^= k >> 16;
//    k *= 0x85ebca6b;
//    k ^= k >> 13;
//    k *= 0xc2b2ae35;
//    k ^= k >> 16;
//    return k & (capacity-1);
//}

__device__ uint64_t hash_v3(uint64_t x) {
    x ^= x >> 27;
    x *= 0x3C79AC492BA7B653UL;
    x ^= x >> 33;
    x *= 0x1C69B3F74AC4AE35UL;
    x ^= x >> 27;
    return x & (capacity_v3-1);
}

__device__ void create_hashtable_v3(KeyValue** table) {
    *table = (KeyValue *) malloc(capacity_v3 * sizeof(KeyValue));
    if(*table != NULL)
        memset(*table, 0xff, sizeof(KeyValue) * capacity_v3);
}

__device__ void insert_v3(KeyValue* hashtable, KeyValue* kv, bool *success) {
    void *key = kv->key;
    void *value = kv->value;
    uint64_t slot = hash_v3((uint64_t) key);
    uint64_t slot_start = slot;
    while (true) {
        void *prev = (void *) atomicCAS((unsigned long long int *)(&hashtable[slot].key),(unsigned long long int) empty, (unsigned long long int) key);
        if (prev == empty || prev == key) {
            hashtable[slot].value = value;
            *success = true;
            return;
        }
        slot = (slot + 1) & (capacity_v3 - 1);
        if(slot == (slot_start + max_chain_v3) % capacity_v3) {
            *success = false;
            return;
        }
    }
}

__device__ void lookup_v3(KeyValue *hashtable, KeyValue *kv) {
    void *key = kv->key;
    uint64_t slot = hash_v3((uint64_t)key);

    while (true) {
        if(hashtable[slot].key == key) {
            kv->value = hashtable[slot].value;
            return;
        } else if(hashtable[slot].key == empty) {
            kv->value = empty;
            return;
        }
        slot = (slot + 1) & (capacity_v3 - 1);
    }
}

// Cannot remove the assigned slot, otherwise all keys that come after ''will get lost''
__device__ void remove_v3(KeyValue *hashtable, KeyValue *kv) {
    void *key = kv->key;
    uint64_t slot = hash_v3((uint64_t)key);

    while (true) {
        if(hashtable[slot].key == key) {
            hashtable[slot].value = empty;
            return;
        } else if(hashtable[slot].key == empty) {
            return;
        }
        slot = (slot + 1) & (capacity_v3 - 1);
    }
}

__device__ void destroy_hashtable_v3(KeyValue **hashtable) {
    free(*hashtable);
    *hashtable = NULL;
}

__shared__ KeyValue *table;
__shared__ void **mem;
//__shared__ int current = 0;

__device__ void init_malloc_v3() {
    if(threadIdx.x == 0) {
        table = (KeyValue *) malloc(capacity_v3 * sizeof(KeyValue));
        if(table != NULL)
            memset(table, 0xff, sizeof(KeyValue) * capacity_v3);
    }
    auto block = cooperative_groups::this_thread_block();
    block.sync();
}

__device__ void clean_malloc() {
    if(threadIdx.x == 0) {
        free(table);
        table = NULL;
    }
    auto block = cooperative_groups::this_thread_block();
    block.sync();
}
__device__ void* malloc_v3(size_t size) {
    /*if(!coalesced) {
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
    }*/

    if (threadIdx.x == 0) {
        //if(!current) {
        //    current = 1;
        //    table = (KeyValue *) malloc(capacity_v3 * sizeof(KeyValue));
        //    if(table != NULL)
        //        memset(table, 0xff, sizeof(KeyValue) * capacity_v3);
        //}
        *mem = malloc(size * blockDim.x + sizeof(int));
        // Initialize counter
        **(int**)mem = blockDim.x;
    }

    auto block = cooperative_groups::this_thread_block();

    block.sync();

    void *ptr = (char*)*mem + sizeof(int) + threadIdx.x * size;
    KeyValue kv = {
        .key = (void *) ptr,
        .value = (void *) *mem
    };

    bool success;
    insert_v3(table, &kv, &success);

    block.sync();

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

    lookup_v3(table, &kv);
    int *counter_ptr = (int *) kv.value;
    int counter = atomicSub(counter_ptr, 1);
    remove_v3(table, &kv);

    if (counter == 1) {
        free(counter_ptr);
    }
}
