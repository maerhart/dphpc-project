#!/bin/bash

set -e 

scriptdir=$(readlink -f $(dirname "$0"))

cd "$scriptdir"

GPU_MPI_BUILD="$scriptdir/../../"
cmake --build "${GPU_MPI_BUILD}"
source "${GPU_MPI_BUILD}/scripts/environment.sh"

echo -n "" > results_nas_is.txt

sed "s,MPICC = mpicc,MPICC = $MPICC,g" ./NPB3.4.1/NPB3.4-MPI/config/make.def.template > ./NPB3.4.1/NPB3.4-MPI/config/make.def
make -C ./NPB3.4.1/NPB3.4-MPI clean
make -C ./NPB3.4.1/NPB3.4-MPI IS CLASS=S
make -C ./NPB3.4.1/NPB3.4-MPI IS CLASS=W
make -C ./NPB3.4.1/NPB3.4-MPI IS CLASS=A
./NPB3.4.1/NPB3.4-MPI/bin/is.S.x ---gpumpi -g 1 -b 1 -p 1000000000 > gpu_is_s_1.txt
./NPB3.4.1/NPB3.4-MPI/bin/is.W.x ---gpumpi -g 4 -b 1 -p 1000000000 > gpu_is_w_4.txt
./NPB3.4.1/NPB3.4-MPI/bin/is.A.x ---gpumpi -g 32 -b 1 -p 1000000000 > gpu_is_a_32.txt

echo -n "GPU IS S 1 " >> results_nas_is.txt
cat gpu_is_s_1.txt | grep "Time in seconds" | awk '{print $5}' >> results_nas_is.txt
echo -n "GPU IS W 4 " >> results_nas_is.txt
cat gpu_is_w_4.txt | grep "Time in seconds" | awk '{print $5}' >> results_nas_is.txt
echo -n "GPU IS A 32 " >> results_nas_is.txt
cat gpu_is_a_32.txt | grep "Time in seconds" | awk '{print $5}' >> results_nas_is.txt


MPICC=mpicc
sed "s,MPICC = mpicc,MPICC = $MPICC,g" ./NPB3.4.1/NPB3.4-MPI/config/make.def.template > ./NPB3.4.1/NPB3.4-MPI/config/make.def
make -C ./NPB3.4.1/NPB3.4-MPI clean
make -C ./NPB3.4.1/NPB3.4-MPI IS CLASS=S
make -C ./NPB3.4.1/NPB3.4-MPI IS CLASS=W
make -C ./NPB3.4.1/NPB3.4-MPI IS CLASS=A
mpirun --oversubscribe -np 1 ./NPB3.4.1/NPB3.4-MPI/bin/is.S.x > cpu_is_s_1.txt
mpirun --oversubscribe -np 4 ./NPB3.4.1/NPB3.4-MPI/bin/is.W.x > cpu_is_w_4.txt
mpirun --oversubscribe -np 32 ./NPB3.4.1/NPB3.4-MPI/bin/is.A.x > cpu_is_a_32.txt


echo -n "CPU IS S 1 " >> results_nas_is.txt
cat cpu_is_s_1.txt | grep "Time in seconds" | awk '{print $5}' >> results_nas_is.txt
echo -n "CPU IS W 4 " >> results_nas_is.txt
cat cpu_is_w_4.txt | grep "Time in seconds" | awk '{print $5}' >> results_nas_is.txt
echo -n "CPU IS A 32 " >> results_nas_is.txt
cat cpu_is_a_32.txt | grep "Time in seconds" | awk '{print $5}' >> results_nas_is.txt
