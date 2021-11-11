#include "gm_malloc.hpp"
#include <iostream>


// do some simple tests
void test()
{
	/* TODO how to do unit tests
	// is_free
	{
		full_header h;
		h.size = 0;
		// h.prev = NULL;

		assert(!is_free(&h));

		h.size = 200;
		assert(!is_free(&h));

		h.size = -1;
		assert(is_free(&h));

		h.size = 0x8000000000000000;
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
	*/

	// repeatedly allocate individual ints and sum them up
	{
		const int MEM_SIZE = 8192;
		const int num_ints = 100; // need enough space for num_ints ints and free list and headers

		// init free list
		gm_malloc::init(10000);

		// repeat test multiple times on the same free_blocks datastructure
		for (int num_runs = 0; num_runs < 10; ++num_runs) {

			// array with results
			int* result[num_ints];
			int* reference[num_ints];


			// fill available space with ints 0, 1, ...
			for (int i = 0; i < num_ints; ++i) {
				// custom malloc
				int* val = (int*)gm_malloc::custom_malloc(sizeof(int));
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
				gm_malloc::custom_free(result[i]);
				free(reference[i]);
			}

		}
		// clean up
		gm_malloc::destroy();
	}

}

int main(int argc, char* argv[]) {
	// run some simple unit tests, only in debug mode!
	test();

	std::cout << "Tests passed" << std::endl;

	gm_malloc::init(10000);
	int* test_int = (int*)gm_malloc::custom_malloc(sizeof(int));
	*test_int = 5;

	std::cout << "Malloc done" << std::endl;

	// clean up
	gm_malloc::destroy();
	return 0;
}

