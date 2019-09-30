#!/bin/bash

set -x
set -e

SCRIPTDIR=$(dirname "$0")

# create ".cuh" for very simple headers, because libtooling skips them
for file in $(find . -name '*\.h'); do
	if [ ! -f "$file.cuh" ]; then
		cp "$file" "$file.cuh"
	fi
done

# add ".cuh" after each filename
files=$(find . -name '*\.cu*')
for file in $files
do
	sed "$file" -i -e 's/\.cuh//g' # remove text if it is already here 
	sed "$file" -i -e 's/#include ["<]\(.*\)[">]/#include "\1.cuh"/g' 
done

# add all files with main functions and create separate binaries for them
main_files=$(grep -rl "__gpu_main_kernel" .)

# add all files without main functions

# files with main functions
# for them we create binaries
files_with_main=$(grep --include "*.cu" -rl "__gpu_main_kernel" .)

# files without main functions
# for them we create object files
files_without_main=$(grep --include "*.cu" -rlL "__gpu_main_kernel" .)

# find list of includes
project_include_dirs=$(find . -type d | sed 's/^/-I/')

mpi_include_dirs=$(mpicc -showme:compile | sed 's/ /\n/g' | grep '^-I')

gpu_mpi_include_dirs=$(find ~/code/gpumpi/gpu_libs/ -type d | sed 's/^/-I/')

gpu_mpi_libraries=$(ls ~/code/build-gpumpi-Desktop-Debug/gpu_libs/*.a)


# for each file we create object file
for f in ${files_without_main} ${files_with_main}; do
	original_file_name=$(echo $f | sed -e 's/\.cuh//g' -e 's/\.cu//g')
	includes=$("$SCRIPTDIR/show_compile_options.py" "${original_file_name}" | grep '^-I')
	defines=$("$SCRIPTDIR/show_compile_options.py" "${original_file_name}" | grep '^-D')
	nvcc $includes $defines ${mpi_include_dirs} ${gpu_mpi_include_dirs} -x cu -dc $f -o $f.o
done

# link each file with main function with other object files without main function
#for f in ${files_without_main}; do
#	nvcc ${project_include_dirs} ${mpi_include_dirs} ${gpu_mpi_include_dirs} -x cu -dc $f -o $f.o
#done
