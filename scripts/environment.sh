#!/bin/bash


SCRIPTDIR=$(readlink -f $(dirname "$BASH_SOURCE"))

export GPU_MPI_PROJECT=$(readlink -f .)
export MPICC=$SCRIPTDIR/gpumpicc.py
export GPU_MPI_MAX_RANKS=$("$SCRIPTDIR/gpu_info" | grep maxCooperativeThreads | cut -d ' ' -f 2)

unset SCRIPTDIR
