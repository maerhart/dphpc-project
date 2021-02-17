#!/bin/bash


SCRIPTDIR=$(readlink -f $(dirname "$BASH_SOURCE"))

export GPU_MPI_PROJECT=$(readlink -f .)
export MPICC=$SCRIPTDIR/gpumpicc.py

unset SCRIPTDIR
