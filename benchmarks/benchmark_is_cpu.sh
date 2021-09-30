#!/bin/bash

set -e

scriptdir=$(readlink -f $(dirname "$0"))

cd "$scriptdir"

bindir="./tmp_run/NPB3.4.1/"

mkdir -p tmp_run

cp -r ../toolchain_tests/npb/NPB3.4.1 "$bindir"

MPICC=mpicc
sed "s,MPICC = mpicc,MPICC = $MPICC,g" "$bindir/NPB3.4-MPI/config/make.def.template" > "$bindir/NPB3.4-MPI/config/make.def"
make -C "$bindir/NPB3.4-MPI" clean
make -C "$bindir/NPB3.4-MPI" IS CLASS=S
make -C "$bindir/NPB3.4-MPI" IS CLASS=W
make -C "$bindir/NPB3.4-MPI" IS CLASS=A
mpirun --oversubscribe -np 1 "$bindir/NPB3.4-MPI/bin/is.S.x" > benchmark_is_cpu_s_1.txt
mpirun --oversubscribe -np 4 "$bindir/NPB3.4-MPI/bin/is.W.x" > benchmark_is_cpu_w_4.txt
mpirun --oversubscribe -np 32 "$bindir/NPB3.4-MPI/bin/is.A.x" > benchmark_is_cpu_a_32.txt

echo -n > benchmark_is_cpu.txt

echo -n "CPU IS S 1 " >> benchmark_is_cpu.txt
cat benchmark_is_cpu_s_1.txt | grep "Time in seconds" | awk '{print $5}' >> benchmark_is_cpu.txt
echo -n "CPU IS W 4 " >> benchmark_is_cpu.txt
cat benchmark_is_cpu_w_4.txt | grep "Time in seconds" | awk '{print $5}' >> benchmark_is_cpu.txt
echo -n "CPU IS A 32 " >> benchmark_is_cpu.txt
cat benchmark_is_cpu_a_32.txt | grep "Time in seconds" | awk '{print $5}' >> benchmark_is_cpu.txt
