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
make -C "$bindir/NPB3.4-MPI" DT CLASS=S
make -C "$bindir/NPB3.4-MPI" DT CLASS=W
make -C "$bindir/NPB3.4-MPI" DT CLASS=A
mpirun --oversubscribe -np 5 "$bindir/NPB3.4-MPI/bin/dt.S.x" BH > benchmark_dt_cpu_s_bh_5.txt
mpirun --oversubscribe -np 11 "$bindir/NPB3.4-MPI/bin/dt.W.x" BH > benchmark_dt_cpu_w_bh_11.txt
mpirun --oversubscribe -np 21 "$bindir/NPB3.4-MPI/bin/dt.A.x" BH > benchmark_dt_cpu_a_bh_21.txt

echo -n > benchmark_dt_cpu.txt

echo -n "CPU DT S BH 5 " >> benchmark_dt_cpu.txt
cat benchmark_dt_cpu_s_bh_5.txt | grep "Time in seconds" | awk '{print $5}' >> benchmark_dt_cpu.txt
echo -n "CPU DT W BH 11 " >> benchmark_dt_cpu.txt
cat benchmark_dt_cpu_w_bh_11.txt | grep "Time in seconds" | awk '{print $5}' >> benchmark_dt_cpu.txt
echo -n "CPU DT A BH 21 " >> benchmark_dt_cpu.txt
cat benchmark_dt_cpu_a_bh_21.txt | grep "Time in seconds" | awk '{print $5}' >> benchmark_dt_cpu.txt
