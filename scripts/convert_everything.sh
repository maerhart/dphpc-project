#!/bin/bash

shopt -s nullglob # if *.c not found, it will be expanded to nothing

make clean && bear make
sources=$(grep "file" compile_commands.json | awk '{ print $2 }' | sed 's/"//g' | sort -u)
echo "SOURCES:" $sources

# 1. sed puts each header file on separate line 
# 2. sed removes slashes
# 3. grep removes all headers that have absolute file paths
headers=$(mpicc -M $sources | sed -e 's/\s/\n/g' -e 's/\\//g' | grep '^[^/].*\.h' | sort -u)
echo "HEADERS:" $headers

~/code/build-gpumpi-Desktop-Debug/_deps/llvm-build/bin/converter $sources $headers -p compile_commands.json 
