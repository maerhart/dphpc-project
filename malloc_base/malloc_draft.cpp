#include <math.h>
#include <iostream>
#include <assert.h>

// header of a free block
struct free_header {
	int64_t size;
	free_header* next;
};

struct full_header {
	// negative means free
	int64_t size;
	// pointer to previous block
	full_header* prev;
};

// sign-bit indicates if block is free
bool is_free(full_header* header) {
	return header->size < 0;
}

// returns appropriate index in free list array according to block size
int get_free_list_index(int64_t size) {
	assert(size >= 1);
	return ceil(log2(size));
}

// size in bytes
void* custom_malloc(size_t size, free_header** free_blocks, const void* range_start, const void* range_end)
{
	// TODO
	return NULL;
}

/*
* @param ptr: block to be freed.
* The parameters below are constant for the whole execution of the allocator.
* @param free_list: points to free blocks datastructure
* @param range_start, range_end: valid range for our memory allocation
*/
void custom_free(void* ptr, free_header** free_list, const void* range_start, const void* range_end)
{
	// check for valid memory range
	if (!(ptr >= range_start && ptr <= range_end)) {
		return;
	}

	// read the header before data starts
	full_header* header = (full_header*)ptr - 1; // -1: points to start of header

	// check if block already free, do nothing
	if (is_free(header)) return;

	// check if next block is free. check if we are still in valid memory range.
	full_header* next_header = (full_header*)((char*)header + sizeof(header) + header->size); // working in bytes
	while (is_free(next_header)
		&& next_header >= (full_header*)range_start
		&& next_header <= (full_header*)range_end)
	{
		// free blocks store header as data
		next_header = (full_header*)((char*)next_header + next_header->size); // working in bytes
	}

	// combine all consecutive free blocks and add to free_list
	free_header* new_header = (free_header*)header;
	new_header->size = (char*)next_header - (char*)header; // gives a positive size => free
	int index = get_free_list_index(new_header->size);
	new_header->next = free_list[index]; // 
	free_list[index] = new_header;

	return;
}

// do some simple tests
void test()
{
	// is_free
	{
		full_header h;
		h.size = 0;
		h.prev = NULL;

		assert(!is_free(&h));

		h.size = 200;
		assert(!is_free(&h));

		h.size = -1;
		assert(is_free(&h));

		h.size = 0x8000000000000000;
		//std::cout << h.size << std::endl;
		assert(is_free(&h));
	}

	// get_free_list_index
	{
		assert(get_free_list_index(1) == 0);
		assert(get_free_list_index(2) == 1);
		assert(get_free_list_index(100) == 7); // 2^6 = 64
	}

}

int main(int argc, char* argv[])
{
	// run some simple unit tests, only in debug mode!
	test();

	int mem_size = 16;

	free_header* initial_block = (free_header*)malloc(mem_size);
	initial_block->size = mem_size;
	initial_block->next = NULL;

	// up to 16 bytes sized blocks for now
	free_header* free_blocks[5];
	free_blocks[4] = initial_block;

	int* test_int = (int*)custom_malloc(sizeof(int), free_blocks, (void*)initial_block, (char*)initial_block + mem_size);
	*test_int = 5;


	return 0;
}