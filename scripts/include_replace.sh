#!/bin/bash

files=$(find . -name '*\.cu*')
for file in $files
do
	echo "$file"
	sed "$file" -i -e 's/#include .\(.*\)./#include "\1.cuh"/g' 
done
