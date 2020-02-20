void* allocate_host_mem(size_t size) {
  // allocate size bytes in a way that we can later send this memory to the host
  // using the delegate_to_host() but also access it from the device
}

void free_host_mem(void* ptr) {
  // free a memory block which was allocated by allocate_host_mem
}

void delegate_to_host(void* mem, size_t size, int mode) {
  // pass mem block of size to host, assume mode is blocking for now
}

void process_gpu_libc(void* mem, size_t size, int mode) {
 // call this function on the host
  // just put comment here where it is

/*
   general_args_t args = mem;
   if (args->calltype == FOPEN) {
   	fopen_args_t* fopen_args = args;
	// do the fopen
   }

}
*/
