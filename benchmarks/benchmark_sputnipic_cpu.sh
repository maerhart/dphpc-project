#!/bin/bash

set -e

scriptdir=$(readlink -f $(dirname "$0"))

cd "$scriptdir"

bindir="./tmp_run/sputnipic/"

mkdir -p tmp_run

cp -r ../toolchain_tests/sputnipic "$bindir"

MPICC=mpicc
sed -i -r "s,param->ncycles = [^;]*,param->ncycles = 5,g;s,param->nxc = [^;]*,param->nxc = 32,g;s,param->nyc = [^;]*,param->nyc = 16,g;s,param->nzc = [^;]*,param->nzc = 16,g" "$bindir/src/Parameters.c"

cd "$bindir"

for f in $(find .  -maxdepth 2 -name "*.cu"); do rm $f; done

make clean
make CXX=$MPICC

cd "$scriptdir"

/usr/bin/time --output benchmark_sputnipic_cpu_1.txt -f "Total runtime in seconds: %e" "$bindir/bin/sputniPIC.out" ---gpumpi -g 1 -p 1000000000 -s 64000 
/usr/bin/time --output benchmark_sputnipic_cpu_2.txt -f "Total runtime in seconds: %e" "$bindir/bin/sputniPIC.out" ---gpumpi -g 2 -p 1000000000 -s 64000 
/usr/bin/time --output benchmark_sputnipic_cpu_4.txt -f "Total runtime in seconds: %e" "$bindir/bin/sputniPIC.out" ---gpumpi -g 4 -p 1000000000 -s 64000 

echo -n > benchmark_sputnipic_cpu.txt

echo -n 'cpu 1 ' >> benchmark_sputnipic_cpu.txt
cat benchmark_sputnipic_cpu_1.txt | grep "Total runtime in seconds" | cut -d ' ' -f 5 >> benchmark_sputnipic_cpu.txt
echo -n 'cpu 2 ' >> benchmark_sputnipic_cpu.txt
cat benchmark_sputnipic_cpu_2.txt | grep "Total runtime in seconds" | cut -d ' ' -f 5 >> benchmark_sputnipic_cpu.txt
echo -n 'cpu 4 ' >> benchmark_sputnipic_cpu.txt
cat benchmark_sputnipic_cpu_4.txt | grep "Total runtime in seconds" | cut -d ' ' -f 5 >> benchmark_sputnipic_cpu.txt
