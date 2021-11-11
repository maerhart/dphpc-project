/*
 * GPU MPI Malloc
 */

#ifndef GM_MALLOC_H
#define GM_MALLOC_H


#include <math.h>
#include <assert.h>
#include <tuple>

namespace gm_malloc {

	// size in bytes
	void* custom_malloc(size_t size);
	
	/*
	* @param ptr: block to be freed.
	* The parameters below are constant for the whole execution of the allocator.
	* @param free_list: points to free blocks datastructure
	* @param range_start, range_end: valid range for our memory allocation
	*/
	void custom_free(void* ptr);

	/**
	 * Initialize the custom gpu mpi malloc
	 *
	 * Initializes free list and memory pool
	 *
	 * @param mem_size Number of bytes to be used (not all available for allocation)
	 */
	 void init(size_t mem_size);

	 /**
	  * Destroy the custom gpu mpi malloc
	  */
	 void destroy();

}

#endif
