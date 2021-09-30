#!/bin/bash

set -e

if [ -z "${GPU_MPI_BUILD}" ]; then
    echo "Please set GPU_MPI_BUILD env var for benchmarking!"
    exit 1
fi

source "${GPU_MPI_BUILD}/scripts/environment.sh"

scriptdir=$(readlink -f $(dirname "$0"))

cd "$scriptdir"

bindir="./tmp_run/sputnipic/"

mkdir -p tmp_run

cp -r ../toolchain_tests/sputnipic "$bindir"

sed -i -r "s,param->ncycles = [^;]*,param->ncycles = 5,g;s,param->nxc = [^;]*,param->nxc = 32,g;s,param->nyc = [^;]*,param->nyc = 16,g;s,param->nzc = [^;]*,param->nzc = 16,g" "$bindir/src/Parameters.c"

cd "$bindir"

for f in $(find .  -maxdepth 2 -name "*.cu"); do rm $f; done

make clean
make CXX=$MPICC

cd "$scriptdir"

/usr/bin/time --output benchmark_sputnipic_gpu_1.txt -f "Total runtime in seconds: %e" "$bindir/bin/sputniPIC.out" ---gpumpi -g 1 -p 1000000000 -s 64000 
/usr/bin/time --output benchmark_sputnipic_gpu_2.txt -f "Total runtime in seconds: %e" "$bindir/bin/sputniPIC.out" ---gpumpi -g 2 -p 1000000000 -s 64000 
/usr/bin/time --output benchmark_sputnipic_gpu_3.txt -f "Total runtime in seconds: %e" "$bindir/bin/sputniPIC.out" ---gpumpi -g 4 -p 1000000000 -s 64000 

echo -n  > benchmark_sputnipic_gpu.txt

echo -n 'gpu 1 ' >> benchmark_sputnipic_gpu.txt
cat benchmark_sputnipic_gpu_1.txt | grep "Total runtime in seconds" | cut -d ' ' -f 5 >> benchmark_sputnipic_gpu.txt
echo -n 'gpu 2 ' >> benchmark_sputnipic_gpu.txt
cat benchmark_sputnipic_gpu_2.txt | grep "Total runtime in seconds" | cut -d ' ' -f 5 >> benchmark_sputnipic_gpu.txt
echo -n 'gpu 4 ' >> benchmark_sputnipic_gpu.txt
cat benchmark_sputnipic_gpu_3.txt | grep "Total runtime in seconds" | cut -d ' ' -f 5 >> benchmark_sputnipic_gpu.txt

