#!/bin/bash

set -e

scriptdir=$(readlink -f $(dirname "$0"))

cd "$scriptdir"

GPU_MPI_BUILD="$scriptdir/../../"
source "${GPU_MPI_BUILD}/scripts/environment.sh"

sed -i -r "s,param->ncycles = [^;]*,param->ncycles = 5,g;s,param->nxc = [^;]*,param->nxc = 32,g;s,param->nyc = [^;]*,param->nyc = 16,g;s,param->nzc = [^;]*,param->nzc = 16,g" src/Parameters.c

for f in $(find .  -maxdepth 2 -name "*.cu"); do rm $f; done

make clean
make CXX=$MPICC

/usr/bin/time --output gpu_sputnipic_1.txt -f "Total runtime in seconds: %e" ./bin/sputniPIC.out ---gpumpi -g 1 -p 1000000000 -s 64000 
/usr/bin/time --output gpu_sputnipic_2.txt -f "Total runtime in seconds: %e" ./bin/sputniPIC.out ---gpumpi -g 2 -p 1000000000 -s 64000 
/usr/bin/time --output gpu_sputnipic_4.txt -f "Total runtime in seconds: %e" ./bin/sputniPIC.out ---gpumpi -g 4 -p 1000000000 -s 64000 

for f in $(find .  -maxdepth 2 -name "*.cu"); do rm $f; done
MPICC=mpicc
make clean
make CXX=$MPICC

export OMP_NUM_THREADS=1

/usr/bin/time --output cpu_sputnipic_1.txt -f "Total runtime in seconds: %e" mpirun -np 1 ./bin/sputniPIC.out
/usr/bin/time --output cpu_sputnipic_2.txt -f "Total runtime in seconds: %e" mpirun -np 2 ./bin/sputniPIC.out
/usr/bin/time --output cpu_sputnipic_4.txt -f "Total runtime in seconds: %e" mpirun -np 4 ./bin/sputniPIC.out

echo -n "" > sputnipic_result.txt
echo -n 'gpu 1 ' >> sputnipic_result.txt
cat gpu_sputnipic_1.txt | grep "Total runtime in seconds" | cut -d ' ' -f 5 >> sputnipic_result.txt
echo -n 'gpu 2 ' >> sputnipic_result.txt
cat gpu_sputnipic_2.txt | grep "Total runtime in seconds" | cut -d ' ' -f 5 >> sputnipic_result.txt
echo -n 'gpu 4 ' >> sputnipic_result.txt
cat gpu_sputnipic_4.txt | grep "Total runtime in seconds" | cut -d ' ' -f 5 >> sputnipic_result.txt

echo -n 'cpu 1 ' >> sputnipic_result.txt
cat cpu_sputnipic_1.txt | grep "Total runtime in seconds" | cut -d ' ' -f 5 >> sputnipic_result.txt
echo -n 'cpu 2 ' >> sputnipic_result.txt
cat cpu_sputnipic_2.txt | grep "Total runtime in seconds" | cut -d ' ' -f 5 >> sputnipic_result.txt
echo -n 'cpu 4 ' >> sputnipic_result.txt
cat cpu_sputnipic_4.txt | grep "Total runtime in seconds" | cut -d ' ' -f 5 >> sputnipic_result.txt
