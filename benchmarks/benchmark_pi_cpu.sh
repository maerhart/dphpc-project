#!/bin/bash

set -e

if [ -z "${MAXRANKS}" ]; then
    echo "Please set MAXRANKS env var for benchmarking!"
    exit 1
fi

scriptdir=$(readlink -f $(dirname "$0"))

cd "$scriptdir"

bindir="./tmp_run/pi/"

mkdir -p tmp_run
cp -r ../toolchain_tests/pi $bindir

cd $bindir
mpicc cpi.c -O3 -o ./pi_cpu

cd "$scriptdir"

echo -n > benchmark_pi_cpu.txt

for (( threads=1; threads<${MAXRANKS}; threads *= 2 ))
do

mpirun -np $threads "$bindir/pi_cpu" 10000000 | tee benchmark_pi_cpu_${threads}.txt
echo -n "cpu $threads " | tee -a benchmark_pi_cpu.txt
cat benchmark_pi_cpu_${threads}.txt | grep "wall clock time" | cut -d ' ' -f 5 | tee -a benchmark_pi_cpu.txt

done

