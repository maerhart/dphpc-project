#include <math.h>
#include <iostream>
#include <assert.h>
#include <tuple>

const int LOG_MIN_BLOCK_SIZE = 3; // allocate at least eight bytes
const size_t MIN_BLOCK_SIZE = pow(2.0, LOG_MIN_BLOCK_SIZE);
const size_t MAX_BLOCK_SIZE = pow(2.0, 32.0); // has to be power of 2 -- allocate at most 4 GB
const int LEN_FREE_LIST = ((int) log2(MAX_BLOCK_SIZE)) - ((int) log2(MIN_BLOCK_SIZE)) + 1;

/* free list design:
 *
 * freelist[i] contains linked list of blocks with payload size of at least 2 ^ (i + log2(MIN_BLOCK_SIZE)) bytes (excluding header size) 
 * where the last block contains null as the next pointer. (i.e. list of size 0 if just a null pointer)
 *
 *
 */


// header of a free block
struct free_header {
	int64_t size; // size of the payload (excluding header size)
	free_header* next;
};

struct full_header {
	// negative means free
	int64_t size; // size of the payload (excluding header size)
	// pointer to previous block
	full_header* prev;
};

// sign-bit indicates if block is free
bool is_free(full_header* header) {
	return header->size < 0;
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
	return floor(log2(size)) - LOG_MIN_BLOCK_SIZE;
}


/**
 * Initialize the free list
 *
 * @param mem_size Number of bytes to be used for first blocks and free list
 * @return pointer to free list
 *         pointer to start of usable memory range
 *         pointer to end of usable memory range
 */
std::tuple<free_header**, void*, void*> init_free_list(size_t mem_size) {

	// check that size big enough
	size_t size_free_list = sizeof(free_header*) * LEN_FREE_LIST;
	if (mem_size < size_free_list + sizeof(full_header) + MIN_BLOCK_SIZE) {
		assert(false); // TODO how to throw
	}

  	// create empty free lists
	free_header** free_list = (free_header**)malloc(size_free_list);  // TODO malloc only once?
	for (int i = 0; i < LEN_FREE_LIST; i++) {
	  free_list[i] = NULL;
	}

	size_t available_mem = mem_size - size_free_list;
  
  	// allocate memory
	free_header* initial_block = (free_header*)malloc(available_mem);

	// build initial block
	size_t payload_size = available_mem - sizeof(full_header);
	initial_block->size = payload_size;
	initial_block->next = NULL;

	// insert block into free list
	free_list[get_free_list_insertion_index(payload_size)] = initial_block;

	// compute range of usable memory
	void* range_start = initial_block;
	void* range_end = ((char*) initial_block) + sizeof(full_header) + payload_size;

	return std::make_tuple(free_list, range_start, range_end);
}


// size in bytes
void* custom_malloc(size_t size, free_header** free_list, const void* range_start, const void* range_end)
{
  	// check if size valid
  	if (size < 1) {
	  assert(false); // TODO how to throw here? or return NULL?
	} else if (size > MAX_BLOCK_SIZE) {
	  assert(false); // TODO how to throw here? or return NULL?
	}

	int free_list_index = get_free_list_extraction_index(size);


  	// TODO check size legal?
	
  
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
	  	assert(false); // fail fast. TODO how to throw here?
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
	int index = get_free_list_insertion_index(new_header->size);
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

	// get_free_list_insertion_index
	{
		assert(get_free_list_insertion_index(MIN_BLOCK_SIZE) == 0);
		assert(get_free_list_insertion_index(MIN_BLOCK_SIZE + 1) == 0);
		assert(get_free_list_insertion_index(MAX_BLOCK_SIZE) == LEN_FREE_LIST - 1);
		assert(get_free_list_insertion_index(MAX_BLOCK_SIZE - 1) == LEN_FREE_LIST - 2);
	}

	// get_free_list_insertion_index
	{
		assert(get_free_list_extraction_index(MIN_BLOCK_SIZE) == 0);
		assert(get_free_list_extraction_index(MIN_BLOCK_SIZE + 1) == 1);
		assert(get_free_list_extraction_index(MAX_BLOCK_SIZE) == LEN_FREE_LIST - 1);
		assert(get_free_list_extraction_index(MAX_BLOCK_SIZE - 1) == LEN_FREE_LIST - 1);
	}

	/*
	// repeatedly allocate individual ints and sum them up
	{
		#define MEM_SIZE 1024
		#define BLOCK_IND 10

		free_header* initial_block = (free_header*)calloc(MEM_SIZE, 1);
		initial_block->size = MEM_SIZE;
		initial_block->next = NULL;

		// put block in free list
		free_header* free_blocks[BLOCK_IND + 1];
		free_blocks[BLOCK_IND] = initial_block;

		// repeat test multiple times on the same free_blocks datastructure
		for (int num_runs = 0; num_runs < 10; ++num_runs) {

			// array with results
			const int num_ints = MEM_SIZE / sizeof(int);
			int* result[num_ints];
			int* reference[num_ints];

			// fill available space with ints 0, 1, ...
			for (int i = 0; i < num_ints; ++i) {
				// custom malloc
				int* val = (int*)custom_malloc(sizeof(int), free_blocks, (void*)initial_block, (char*)initial_block + MEM_SIZE);
				assert(val); // No null pointers
				*val = i;
				result[i] = val;

				// reference
				int* val_ref = (int*)malloc(sizeof(int));
				assert(val_ref); // No null pointers
				*val_ref = i;
				reference[i] = val_ref;
			}

			// compare results
			int res_sum = 0;
			int ref_sum = 0;
			for (int i = 0; i < num_ints; ++i) {

				res_sum += *(result[i]);
				ref_sum += *(reference[i]);

				// correct value stored
				assert(*(result[i]) == i);
				assert(*(reference[i]) == i);
			}

			// correct sum
			assert(ref_sum == (num_ints - 1) * (num_ints) / 2); // analytic solution
			assert(res_sum == ref_sum); // reference solution


			// free for the next round
			for (int i = 0; i < num_ints; ++i) {
				custom_free(result[i], free_blocks, (void*)initial_block, (char*)initial_block + MEM_SIZE);
				free(reference[i]);
			}

		}


		// clean up
		free(initial_block);
	}
	*/

}

int main(int argc, char* argv[])
{
	// run some simple unit tests, only in debug mode!
	test();

	std::tuple<free_header**, void*, void*> init_data = init_free_list(10000); // contains free list, range_start, range_end


	int* test_int = (int*)custom_malloc(sizeof(int), std::get<0>(init_data), std::get<1>(init_data), std::get<2>(init_data));
	//*test_int = 5;

	return 0;
}
