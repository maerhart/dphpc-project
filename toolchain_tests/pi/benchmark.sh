#!/bin/bash

set -e

scriptdir=$(readlink -f $(dirname "$0"))

cd "$scriptdir"

GPU_MPI_BUILD="$scriptdir/../../"

source "${GPU_MPI_BUILD}/scripts/environment.sh"
$MPICC cpi.c -o ./pi_gpu
mpicc cpi.c -O3 -o ./pi_cpu

if [ -z "${GPU_MPI_MAX_RANKS}" ]; then
    echo "ERROR! GPU_MPI_MAX_RANKS is unset!"
    exit 1
fi

echo -n > plot_pi_gpu.txt
echo -n > plot_pi_cpu.txt

for (( threads=1; threads<${GPU_MPI_MAX_RANKS}; threads *= 2 ))
do

./pi_gpu 10000000 ---gpumpi -s 8192 -n $threads | tee output_pi_gpu_${threads}.txt
echo -n "gpu $threads " | tee -a plot_pi_gpu.txt
cat output_pi_gpu_${threads}.txt | grep "wall clock time" | cut -d ' ' -f 5 | tee -a plot_pi_gpu.txt

mpirun -np $threads ./pi_cpu 10000000 | tee output_pi_cpu_${threads}.txt
echo -n "cpu $threads " | tee -a plot_pi_cpu.txt
cat output_pi_cpu_${threads}.txt | grep "wall clock time" | cut -d ' ' -f 5 | tee -a plot_pi_cpu.txt

done

