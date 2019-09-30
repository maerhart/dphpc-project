#!/bin/bash

set -x
set -e

rm -rf ./gpumpi_build

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

#gpu_mpi_include_dirs=$(find ~/code/gpumpi/gpu_libs/ -type d | sed 's/^/-I/')
gpu_mpi_include_dirs=$(find ~/code/gpumpi/gpu_libs/ -type d)

gpu_mpi_libraries=$(ls ~/code/build-gpumpi-Desktop-Debug/gpu_libs/*.a)

cat << EOF > CMakeLists.txt 
cmake_minimum_required(VERSION 3.12)

add_subdirectory("${SCRIPTDIR}/../gpu_libs" "gpu_libs")

project(examples LANGUAGES C CXX CUDA)

set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
EOF

for file in ${files_without_main}; do

original_file_name=$(echo ${file} | sed -e 's/\.cuh//g' -e 's/\.cu//g')
includes=$("$SCRIPTDIR/show_compile_options.py" "${original_file_name}" | grep '^-I' | sed 's/^-I//g')
defines=$("$SCRIPTDIR/show_compile_options.py" "${original_file_name}" | grep '^-D' | sed 's/^-D//g')
target=target_lib_$(echo ${file} | tr '/' '_' | tr '.' '_')

cat << EOF >> CMakeLists.txt
add_library(${target} ${file})
target_include_directories(${target} PRIVATE ${includes} ${gpu_mpi_include_dirs})
EOF

done

for file in ${files_with_main}; do


original_file_name=$(echo ${file} | sed -e 's/\.cuh//g' -e 's/\.cu//g')
includes=$("$SCRIPTDIR/show_compile_options.py" "${original_file_name}" | grep '^-I' | sed 's/^-I//g')
defines=$("$SCRIPTDIR/show_compile_options.py" "${original_file_name}" | grep '^-D' | sed 's/^-D//g')
target=target_bin_$(echo ${file} | tr '/' '_' | tr '.' '_')

cat << EOF >> CMakeLists.txt
add_executable(${target} ${file})
target_include_directories(${target} PRIVATE ${includes} ${gpu_mpi_include_dirs})
target_link_libraries(${target} PRIVATE gpu_libs)
EOF

for lib in ${files_without_main}; do

lib_target=target_lib_$(echo ${lib} | tr '/' '_' | tr '.' '_')

cat << EOF >> CMakeLists.txt
target_link_libraries(${target} PRIVATE ${lib_target})
EOF

done


done

# # for each file we create object file
# for f in ${files_without_main} ${files_with_main}; do
# 	original_file_name=$(echo $f | sed -e 's/\.cuh//g' -e 's/\.cu//g')
# 	includes=$("$SCRIPTDIR/show_compile_options.py" "${original_file_name}" | grep '^-I')
# 	defines=$("$SCRIPTDIR/show_compile_options.py" "${original_file_name}" | grep '^-D')
# 	nvcc $includes $defines ${mpi_include_dirs} ${gpu_mpi_include_dirs} -x cu -dc $f -o $f.o
# done
# 
# # link each file with main function with other object files without main function
# for f in ${files_with_main}; do
# 	nvcc ${project_include_dirs} ${mpi_include_dirs} ${gpu_mpi_include_dirs} -x cu -dc $f -o $f.o
# done

mkdir ./gpumpi_build
cd ./gpumpi_build
cmake ..
make VERBOSE=1
