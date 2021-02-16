#!/bin/bash

scriptdir=$(readlink -f $(dirname "$0"))

cd "$scriptdir"

export GPU_MPI_PROJECT=$(readlink -f .)
export MPICC=../../scripts/gpumpicc.py

make

./global_var ---gpumpi -g 4 -b 1
