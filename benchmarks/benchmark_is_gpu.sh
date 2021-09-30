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
make -C "$bindir/NPB3.4-MPI" IS CLASS=S
make -C "$bindir/NPB3.4-MPI" IS CLASS=W
make -C "$bindir/NPB3.4-MPI" IS CLASS=A
"$bindir/NPB3.4-MPI/bin/is.S.x" ---gpumpi -g 1 -b 1 -p 1000000000 > benchmark_is_gpu_s_1.txt
"$bindir/NPB3.4-MPI/bin/is.W.x" ---gpumpi -g 4 -b 1 -p 1000000000 > benchmark_is_gpu_w_4.txt
"$bindir/NPB3.4-MPI/bin/is.A.x" ---gpumpi -g 32 -b 1 -p 1000000000 > benchmark_is_gpu_a_32.txt

echo -n > benchmark_is_gpu.txt

echo -n "GPU IS S 1 " >> benchmark_is_gpu.txt
cat benchmark_is_gpu_s_1.txt | grep "Time in seconds" | awk '{print $5}' >> benchmark_is_gpu.txt
echo -n "GPU IS W 4 " >> benchmark_is_gpu.txt
cat benchmark_is_gpu_w_4.txt | grep "Time in seconds" | awk '{print $5}' >> benchmark_is_gpu.txt
echo -n "GPU IS A 32 " >> benchmark_is_gpu.txt
cat benchmark_is_gpu_a_32.txt | grep "Time in seconds" | awk '{print $5}' >> benchmark_is_gpu.txt

