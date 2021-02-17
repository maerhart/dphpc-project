#!/bin/bash

SCRIPTDIR=$(readlink -f $(dirname "$0"))

cd "$SCRIPTDIR"

export GPU_MPI_PROJECT="$SCRIPTDIR/NPB3.4.1"

export MPICC=$(readlink -f ../../scripts/gpumpicc.py)

# use comma as delimiter in sed command to avoid collision with slash in the path
sed "s,MPICC = mpicc,MPICC = $MPICC,g" $SCRIPTDIR/NPB3.4.1/NPB3.4-MPI/config/make.def.template > $SCRIPTDIR/NPB3.4.1/NPB3.4-MPI/config/make.def

cd $SCRIPTDIR/NPB3.4.1/NPB3.4-MPI

make IS CLASS=S

./bin/is.S.x ---gpumpi -g 2 -b 1
