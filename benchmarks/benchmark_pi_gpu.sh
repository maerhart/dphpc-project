#!/bin/bash

set -e

if [ -z "${GPU_MPI_BUILD}" ]; then
    echo "Please set GPU_MPI_BUILD env var for benchmarking!"
    exit 1
fi

source "${GPU_MPI_BUILD}/scripts/environment.sh"

scriptdir=$(readlink -f $(dirname "$0"))

cd "$scriptdir"

bindir="./tmp_run/pi/"

mkdir -p tmp_run
cp -r ../toolchain_tests/pi $bindir

cd $bindir
$MPICC cpi.c -o ./pi_gpu

cd "$scriptdir"

echo -n > benchmark_pi_gpu.txt

for (( threads=1; threads<${GPU_MPI_MAX_RANKS}; threads *= 2 ))
do

"$bindir/pi_gpu" 10000000 ---gpumpi -s 8192 -n $threads | tee benchmark_pi_gpu_${threads}.txt
echo -n "gpu $threads " | tee -a benchmark_pi_gpu.txt
cat benchmark_pi_gpu_${threads}.txt | grep "wall clock time" | cut -d ' ' -f 5 | tee -a benchmark_pi_gpu.txt

done

