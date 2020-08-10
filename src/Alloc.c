#include <Alloc.h>
#include <string.h>

void *_newArr(int typesize, int dim, size_t sz1, va_list sizeargs){

	void** arr;
	if(dim < 2){
		arr = (void**) malloc(sz1*typesize);
        if (!arr) {
            printf("Malloc can't allocate memory, aborting...\n");
            abort();
        }
		memset((void*) arr, 0, sz1*typesize);
		return (void*) arr;
	}
	else{
		arr = (void**) malloc(sz1*sizeof(void*));
        if (!arr) {
            printf("Malloc can't allocate memory, aborting...\n");
            abort();
        }
	}

	size_t sz2 = va_arg(sizeargs, size_t);
	void* ptr = _newArr(typesize, dim-1, sz1*sz2, sizeargs);

	int size = typesize;
	if(dim > 2){
		size = sizeof(void*);
	}
	for (size_t i = 0; i < sz1; i++) {
		arr[i] = ptr;
		ptr = ((char*)ptr) + sz2*size;
	}
	return (void*) arr;
}

/**
 * Create a d dimensional pointer hierarchy array
 */
void *newArr(int typesize, int dim, ...){

	va_list args;
	va_start(args, dim);
	size_t sz1 = va_arg(args, size_t);
	void* arr = _newArr(typesize, dim, sz1, args);
	va_end(args);

	return arr;
}

/**
 * Create hierarchical pointer structure to mimic multidimensional arrays, 
 * where data is dynamically allocated on the heap, and data array is 
 * contigous in memory to allow efficient communication of data. 
 */
void **ptrArr(void **in, int typesize, int dim, ...){

	// Initialize variadic arguments
	va_list args, args_cpy;
	va_start(args, dim);
	va_copy(args_cpy, args);

	// Calculate combined sizes
	size_t szarr = 1;
	for(int i=0; i<dim-1; i++){
		szarr *= va_arg(args, size_t);
	}
	size_t sz0 = va_arg(args, size_t);

	// Array to hold data
	*in = newArr(typesize, 1, szarr*sz0);
	void *ptr = *in;

	// Create hierarchy of pointer to pointer arrays, d-1 deep.
	size_t sz1 = va_arg(args_cpy, size_t);
	void **arr = (void**)_newArr(sizeof(void*), dim-1, sz1, args_cpy);
	void **arr2 = arr;

	// Dereference pointer to array d-2 times
	for(int i=0; i<dim-2; i++){
		arr2 = (void**) *arr2;
	}

	// Set pointer location for bottom pointer array to point to data
	for(size_t i=0; i<szarr; i++){
		arr2[i] = ptr;
		ptr = ((char*)ptr) + sz0*typesize;
	}

	va_end(args);
	return arr;
}

void delArr(int dim, void *arr) {
	if(dim > 1)
		delArr(dim-1, ((void**)arr)[0]);
	free(arr);
}
