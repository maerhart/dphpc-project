#!/bin/bash

# add ".cuh" after each filename
files=$(find . -name '*\.cu*')
for file in $files
do
	echo "$file"
	sed "$file" -i -e 's/#include .\(.*\)./#include "\1.cuh"/g' 
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



