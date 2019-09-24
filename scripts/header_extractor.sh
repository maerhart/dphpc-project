#!/bin/bash

# prints list of local headers included by source file provided as argument
# usage: header_extractor.sh file.c
# 1. sed puts each header file on separate line 
# 2. sed removes slashes
# 3. grep removes all headers that have absolute file paths

mpicc -M $@ | sed -e 's/\s/\n/g' -e 's/\\//g' | grep '^[^/].*\.h' | sort -u
