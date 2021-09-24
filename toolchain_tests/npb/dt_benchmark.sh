#!/bin/bash

set -e 

scriptdir=$(readlink -f $(dirname "$0"))

cd "$scriptdir"

GPU_MPI_BUILD="$scriptdir/../../"
source "${GPU_MPI_BUILD}/scripts/environment.sh"

echo -n "" > results_nas_dt.txt

sed "s,MPICC = mpicc,MPICC = $MPICC,g" ./NPB3.4.1/NPB3.4-MPI/config/make.def.template > ./NPB3.4.1/NPB3.4-MPI/config/make.def
make -C ./NPB3.4.1/NPB3.4-MPI clean
make -C ./NPB3.4.1/NPB3.4-MPI DT CLASS=S
make -C ./NPB3.4.1/NPB3.4-MPI DT CLASS=W
make -C ./NPB3.4.1/NPB3.4-MPI DT CLASS=A
./NPB3.4.1/NPB3.4-MPI/bin/dt.S.x BH ---gpumpi -g 5 -b 1 -s 16000 -p 640000000 > gpu_dt_s_bh_5.txt
./NPB3.4.1/NPB3.4-MPI/bin/dt.W.x BH ---gpumpi -g 11 -b 1 -s 16000 -p 640000000 > gpu_dt_w_bh_11.txt
./NPB3.4.1/NPB3.4-MPI/bin/dt.A.x BH ---gpumpi -g 21 -b 1 -s 16000 -p 640000000 > gpu_dt_a_bh_21.txt

echo -n "GPU DT S BH 5 " >> results_nas_dt.txt
cat gpu_dt_s_bh_5.txt | grep "Time in seconds" | awk '{print $5}' >> results_nas_dt.txt
echo -n "GPU DT W BH 11 " >> results_nas_dt.txt
cat gpu_dt_w_bh_11.txt | grep "Time in seconds" | awk '{print $5}' >> results_nas_dt.txt
echo -n "GPU DT A BH 21 " >> results_nas_dt.txt
cat gpu_dt_a_bh_21.txt | grep "Time in seconds" | awk '{print $5}' >> results_nas_dt.txt


MPICC=mpicc
sed "s,MPICC = mpicc,MPICC = $MPICC,g" ./NPB3.4.1/NPB3.4-MPI/config/make.def.template > ./NPB3.4.1/NPB3.4-MPI/config/make.def
make -C ./NPB3.4.1/NPB3.4-MPI clean
make -C ./NPB3.4.1/NPB3.4-MPI DT CLASS=S
make -C ./NPB3.4.1/NPB3.4-MPI DT CLASS=W
make -C ./NPB3.4.1/NPB3.4-MPI DT CLASS=A
mpirun --oversubscribe -np 5 ./NPB3.4.1/NPB3.4-MPI/bin/dt.S.x BH > cpu_dt_s_bh_5.txt
mpirun --oversubscribe -np 11 ./NPB3.4.1/NPB3.4-MPI/bin/dt.W.x BH > cpu_dt_w_bh_11.txt
mpirun --oversubscribe -np 21 ./NPB3.4.1/NPB3.4-MPI/bin/dt.A.x BH > cpu_dt_a_bh_21.txt


echo -n "CPU DT S BH 5 " >> results_nas_dt.txt
cat cpu_dt_s_bh_5.txt | grep "Time in seconds" | awk '{print $5}' >> results_nas_dt.txt
echo -n "CPU DT W BH 11 " >> results_nas_dt.txt
cat cpu_dt_w_bh_11.txt | grep "Time in seconds" | awk '{print $5}' >> results_nas_dt.txt
echo -n "CPU DT A BH 21 " >> results_nas_dt.txt
cat cpu_dt_a_bh_21.txt | grep "Time in seconds" | awk '{print $5}' >> results_nas_dt.txt
