#include "gm_malloc.hpp"

namespace gm_malloc {
	/* free list design:
	 *
	 * freelist[i] contains linked list of blocks with payload size of at least 2 ^ (i + log2(MIN_BLOCK_SIZE)) bytes (excluding header size)
	 * where the last block contains null as the next pointer. (i.e. list of size 0 if just a null pointer)
	 *
	 */
	
	// TODO alignment requirements?
	

	// header of a free block
	struct free_header {
		int64_t size; // size of the payload (excluding header size of full_header)
		free_header* next;
		free_header** prev_next_ptr;
	};

	struct full_header {
		// negative means free
		int64_t size; // size of the payload (excluding header size)
	
		/* planned to potentially introduce for better freeing:
		 * full_header* prev;
		 */
	};



	// free list constants
	const int LOG_MIN_BLOCK_SIZE = 4; // allocate at least 16 bytes
	const size_t MIN_BLOCK_SIZE = pow(2.0, LOG_MIN_BLOCK_SIZE); // has to be of at least size sizeof(free_header) - sizeof(full_header)
	const size_t MAX_BLOCK_SIZE = pow(2.0, 32.0); // has to be power of 2 -- allocate at most 4 GB
	const int LEN_FREE_LIST = ((int) log2(MAX_BLOCK_SIZE)) - ((int) log2(MIN_BLOCK_SIZE)) + 1;

	// constants for list headers
	const int64_t SIZE_MASK = ~(((int64_t) 1) << 63);

	// constants for allocated memory
	bool INITIALIZED = false;
	free_header** FREE_LIST = NULL;
	void* RANGE_START = NULL;
	void* RANGE_END = NULL;



	/*
	 * determine whether pointer points to free header or full header
	 *
	 * @param header Pointer to legal header
	 * @return True iff points to free header
	 */
	bool is_free(void* header) {
		assert(((full_header*) header)->size == (*((int64_t*)header))); // TODO remove
		return ((*((int64_t*) header)) & ~SIZE_MASK) != 0; // first bit of size field 1 means block is free
	}
	
	/*
	 * determine whether pointer points to free header or full header
	 *
	 * @param header Pointer to legal header
	 * @return payload size of block header points to
	 * 	(total block size including header - sizeof(full_header))
	 */
	size_t get_size(void* header) {
		assert(((full_header*) header)->size == (*((int64_t*)header))); // TODO remove
		return (*((int64_t*) header)) & SIZE_MASK;
	}
	
	/**
	 * Set block size
	 *
	 * @param header The header of the block
	 * @param size The payload size of the block
	 * 		requires size < 2^63
	 */
	void set_size(full_header* header, size_t size) {
		header->size = ((int64_t) size) | (header->size & ~SIZE_MASK);
	}
	
	/**
	 * Set block size
	 *
	 * @param header The header of the block
	 * @param size The payload size of the block
	 * 		requires size < 2^63
	 */
	void set_size(free_header* header, size_t size) {
		header->size = ((int64_t) size) | (header->size & ~SIZE_MASK);
	}
	
	/**
	 * Make a free block
	 *
	 * @param ptr Pointer to memory location where to set up free block
	 * @param size Payload size of the free block
	 * @return Pointer to the free block
	 */
	free_header* make_free_block(void* ptr, size_t size) {
		free_header* header = (free_header*) ptr;
		// set free bit to 1
		header->size = ((int64_t) 1) << 63;
		set_size(header, size);
		header->next = NULL;
		header->prev_next_ptr = NULL;
		return header;
	}
	
	/**
	 * Make a full/allocated block
	 *
	 * @param ptr Pointer to memory location where to set up block
	 * @param size Payload size of the block
	 * @return Pointer to the block
	 */
	full_header* make_full_block(void* ptr, size_t size) {
		full_header* header = (full_header*) ptr;
		// set free bit to 0
		header->size = 0;
		set_size(header, size);
		return header;
	}
	
	
	/**
	 * Retrieve index into free list for extracting a block of at least the given size
	 *
	 * @param size Size in bytes of the block to be allocated. Not including header size
	 *             requires 1 <= size <= MAX_BLOCK_SIZE
	 * @return index into free list where blocks of at least the given size can be found
	 */
	int get_free_list_extraction_index(int64_t size) {
	  	if (size < 1 || size > MAX_BLOCK_SIZE) {
			assert(false); // TODO how to throw here?
		}
		int ceil_log = ceil(log2(size));
		return std::max(0, ceil_log - LOG_MIN_BLOCK_SIZE);
	}
	
	/**
	 * Retrieve index into free list where to insert a block of the given size
	 *
	 * @param size Size in bytes of the block to be inserted. Not including header size
	 *             requires  size >= MIN_BLOCK_SIZE
	 * @return index into free list where blocks of the given size should be inserted
	 */
	int get_free_list_insertion_index(int64_t size) {
	  	if (size < MIN_BLOCK_SIZE) {
			assert(false); // TODO how to throw here?
		}
		int insertion_index = ((int) log2(size)) - LOG_MIN_BLOCK_SIZE;
		return std::min(insertion_index, LEN_FREE_LIST - 1);
	}
	
	/**
	 * Insert a free block into the free list of the appropriate size
	 *
	 * @param free_list The array of free lists
	 * @param header Pointer to a valid free block
	 */
	void insert_in_free_list(free_header** free_list, free_header* header) {
		assert(is_free(header));
		int index = get_free_list_insertion_index(get_size(header));
		header->next = free_list[index];
		free_list[index] = header;
	
		// set prev_next_ptr  for header and block after
		header->prev_next_ptr = &free_list[index];
		if (header->next != NULL) {
			header->next->prev_next_ptr = &(header->next);
		}
	}
	
	
	/**
	 * Remove a free block from the free list
	 *
	 * @param header Pointer to a valid free block that is contained in a free list
	 */
	void remove_from_free_list(free_header* header) {
		if (header->prev_next_ptr != NULL) {
			*(header->prev_next_ptr) = header->next;
		}
		if (header->next != NULL) {
			header->next->prev_next_ptr = header->prev_next_ptr;
		}
	}
	
	/**
	 * Pop a block from a free list and make it a full block
	 *
	 * @param free_list The array of free lists
	 * @param index Index specifying from which free list (/block size) to pop
	 * 		requires that free_list at index is non empty
	 *
	 * @return The first free block from the specified list as a usable full block
	 */
	full_header* pop_free_list(free_header** free_list, int index) {
		assert(free_list[index] != NULL);
		assert(index >= 0 && index < LEN_FREE_LIST);
	
		free_header* block_free = free_list[index];
	
		remove_from_free_list(block_free);
	
		return make_full_block(block_free, get_size(block_free));
	}
	
	void init(size_t mem_size) {
	
		// check that size big enough
		size_t size_free_list = sizeof(free_header*) * LEN_FREE_LIST;
		if (mem_size < size_free_list + sizeof(full_header) + MIN_BLOCK_SIZE) {
			assert(false); // TODO how to throw
		}
	
	  	// create empty free lists
		// malloc entire memory chunk used for free list and malloc at once
		FREE_LIST = (free_header**)malloc(mem_size);
		for (int i = 0; i < LEN_FREE_LIST; i++) {
		  FREE_LIST[i] = NULL;
		}
	
		size_t available_mem = mem_size - size_free_list;
	  
	  	// give rest of memory to initial block
		char* ptr_initial_block = ((char*)FREE_LIST) + size_free_list;
		size_t payload_size = available_mem - sizeof(full_header);
		free_header* initial_block = make_free_block(ptr_initial_block, payload_size);
		// insert block into free list
		insert_in_free_list(FREE_LIST, initial_block);
	
		// compute range of usable memory
		RANGE_START = initial_block;
		RANGE_END = ((char*) initial_block) + sizeof(full_header) + payload_size;
		
		INITIALIZED = true;
	}
	
	void* custom_malloc(size_t size) {
		assert(INITIALIZED);

	  	// check if size valid
	  	if (size < 1) {
		  assert(false); // TODO how to throw here? or return NULL?
		} else if (size > MAX_BLOCK_SIZE) {
		  assert(false); // TODO how to throw here? or return NULL?
		}
	
		// round up size to min block size
		size = MIN_BLOCK_SIZE * ceil((1.0 * size)/MIN_BLOCK_SIZE);
	
		int free_list_index = get_free_list_extraction_index(size);
	
	
		// find list with free block
		while (FREE_LIST[free_list_index] == NULL) {
			free_list_index++;
			if (free_list_index == LEN_FREE_LIST) {
		  		assert("out of memory" && false); // TODO how to throw here? or return NULL?
			}
		}
	
		// retrieve block
		full_header* block = pop_free_list(FREE_LIST, free_list_index);
		size_t available_payload_size = get_size(block);
	
	
		// split block if necessary
		if (available_payload_size >= sizeof(full_header) + MIN_BLOCK_SIZE + size) {
			// split block
			
			// reduce size of old block
			set_size(block, size);
	
			// create new block
			free_header* new_block = make_free_block(
				((char*) block) + sizeof(full_header) + size, 		// ptr to start of free area
				available_payload_size - size - sizeof(full_header)	// size: need to accomodate new_blocks header and payload of allocated block
			);
	
			insert_in_free_list(FREE_LIST, new_block);
		}
		
		return ((char*)block) + sizeof(full_header);
	}
	
	void custom_free(void* ptr) {
		assert(INITIALIZED);

		// check for valid memory range
		if (!(ptr >= RANGE_START && ptr <= RANGE_END)) {
		  	assert(false); // fail fast. TODO how to throw here?
		}
	
		// read the header before data starts
		full_header* header = ((full_header*)ptr) - 1; // -1: points to start of header
	
		// check if block already free
		if (is_free(header)) {
			assert(false); // freeing twice is undefined behaviour  TODO how to throw here
		}
	
		// check if next block is free. check if we are still in valid memory range.
		char* next_header = ((char*)header) + sizeof(full_header) + get_size(header); // working in bytes
		while (is_free(next_header)
			&& next_header >= (char*)RANGE_START
			&& next_header <= (char*)RANGE_END)
		{
			// the following block is free so we want to merge it with our freed block
			// -> remove from its free list as it will be part of the new bigger free block
			
			// remove from free list
			remove_from_free_list((free_header*) next_header);
	
			// size always specifies payload size that block can fit if full
			next_header = next_header + sizeof(full_header) + get_size(next_header);
		}
	
		// combine all consecutive free blocks and add to free_list
		free_header* freed_block = make_free_block(header, next_header - ((char*)header) - sizeof(full_header));
		insert_in_free_list(FREE_LIST, freed_block);
	}

	void destroy() {
		free(FREE_LIST);
		INITIALIZED = false;
		FREE_LIST = NULL;
		RANGE_START = NULL;
		RANGE_END = NULL;
	}
}



