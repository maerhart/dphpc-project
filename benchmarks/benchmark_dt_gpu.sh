#!/bin/bash

set -e 

if [ -z "${GPU_MPI_BUILD}" ]; then
    echo "Please set GPU_MPI_BUILD env var for benchmarking!"
    exit 1
fi

source "${GPU_MPI_BUILD}/scripts/environment.sh"

scriptdir=$(readlink -f $(dirname "$0"))

cd "$scriptdir"

bindir="./tmp_run/NPB3.4.1/"

mkdir -p tmp_run

cp -r ../toolchain_tests/npb/NPB3.4.1 "$bindir"

sed "s,MPICC = mpicc,MPICC = $MPICC,g" "$bindir/NPB3.4-MPI/config/make.def.template" > "$bindir/NPB3.4-MPI/config/make.def"
make -C "$bindir/NPB3.4-MPI" clean
make -C "$bindir/NPB3.4-MPI" DT CLASS=S
make -C "$bindir/NPB3.4-MPI" DT CLASS=W
make -C "$bindir/NPB3.4-MPI" DT CLASS=A
"$bindir/NPB3.4-MPI/bin/dt.S.x" BH ---gpumpi -g 5 -b 1 -s 16000 -p 640000000 > benchmark_dt_gpu_s_bh_5.txt
"$bindir/NPB3.4-MPI/bin/dt.W.x" BH ---gpumpi -g 11 -b 1 -s 16000 -p 640000000 > benchmark_dt_gpu_w_bh_11.txt
"$bindir/NPB3.4-MPI/bin/dt.A.x" BH ---gpumpi -g 21 -b 1 -s 16000 -p 640000000 > benchmark_dt_gpu_a_bh_21.txt

echo -n > benchmark_dt_gpu.txt

echo -n "GPU DT S BH 5 " >> benchmark_dt_gpu.txt
cat benchmark_dt_gpu_s_bh_5.txt | grep "Time in seconds" | awk '{print $5}' >> benchmark_dt_gpu.txt
echo -n "GPU DT W BH 11 " >> benchmark_dt_gpu.txt
cat benchmark_dt_gpu_w_bh_11.txt | grep "Time in seconds" | awk '{print $5}' >> benchmark_dt_gpu.txt
echo -n "GPU DT A BH 21 " >> benchmark_dt_gpu.txt
cat benchmark_dt_gpu_a_bh_21.txt | grep "Time in seconds" | awk '{print $5}' >> benchmark_dt_gpu.txt


