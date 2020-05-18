# Headers extensions

Files with extension `.cuh` are supposed to contain implementations of 
std library functions on GPU with `__gpu_` prefix.
Files with extension `.h.cuh` map names of std library function to their gpu equivalent and
supposed to be included from converted user code, but not from this library implementation.
