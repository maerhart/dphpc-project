#!/bin/bash

SCRIPTDIR=$(readlink -f $(dirname "$0"))

cd "$SCRIPTDIR"

export GPU_MPI_PROJECT="$SCRIPTDIR/NPB3.4.1"

export MPICC=$(readlink -f ../../scripts/gpumpicc.py)

# use comma as delimiter in sed command to avoid collision with slash in the path
sed "s,MPICC = mpicc,MPICC = $MPICC,g" $SCRIPTDIR/NPB3.4.1/NPB3.4-MPI/config/make.def.template > $SCRIPTDIR/NPB3.4.1/NPB3.4-MPI/config/make.def

cd $SCRIPTDIR/NPB3.4.1/NPB3.4-MPI

make DT CLASS=S

./bin/dt.S.x BH ---gpumpi -g 5 -b 1 -s 16000 -p 80000000

