COMMIT=$(git log -n1 | head -n 1 | cut -c 8-)
DATE=$(date "+%F-%T")
SCRIPT="${BASH_SOURCE[0]:-$0}"
ROOTDIR="$(cd "$(dirname ${SCRIPT})/" && pwd)"
DIRECTORY=$ROOTDIR/results/$DATE #--$COMMIT
BUILD=$ROOTDIR/../build
mkdir -p $DIRECTORY

HEADER="version workload run blocks threads doubles malloc work free"
WORKLOADS="reduce bcast scalar_product combined"
VERSION="baseline v1 v2 v3 v4 v5_anton v1_martin combined_malloc"
MALLOC_ARG=(0 1 2 3 4 5 6 7)
BLOCKS=(240)# 120 60 30)
THREADS=(64)# 128 256 512)

ALLTOALL_BLOCKS=8
ALLTOALL_THREADS=32

RUNS=1

cd $BUILD
echo $HEADER >> $DIRECTORY/out.csv
for version in $VERSION
do
    GPU_MPI_PROJECT=$ROOTDIR/.. GPU_MPI_MAX_RANKS=2000000 python scripts/gpumpicc.py ../mpi_malloc_benchmarks/alltoall.c --malloc_version=${MALLOC_ARG[n]}
for floats in `seq 4 2 5`
do
for runs in `seq 1 1 $RUNS`
do
	echo "$(date) -- $version alltoall $ALLTOALL_BLOCKS $ALLTOALL_THREADS $floats $runs"
	echo "$(date) -- $version alltoall $ALLTOALL_BLOCKS $ALLTOALL_THREADS $floats $runs" >> $DIRECTORY/log.txt
	res=$(timeout 15m ./a.out $floats ---gpumpi -g $ALLTOALL_BLOCKS -b $ALLTOALL_THREADS 2>> $DIRECTORY/log.txt)
	echo "$res" | while read line; do echo "$version alltoall $runs $ALLTOALL_BLOCKS $ALLTOALL_THREADS $floats $line"; done >> $DIRECTORY/out.csv
done
done
done

for version in $VERSION
do
n=0
for workload in $WORKLOADS
do
    GPU_MPI_PROJECT=$ROOTDIR/.. GPU_MPI_MAX_RANKS=2000000 python scripts/gpumpicc.py ../mpi_malloc_benchmarks/$workload.c --malloc_version=${MALLOC_ARG[n]}
	n=$(($n+1))
for i in ${!BLOCKS[@]}
do
for floats in `seq 4 2 5`
do
for runs in `seq 1 1 $RUNS`
do
	echo "$(date) -- $version $workload ${BLOCKS[$i]} ${THREADS[$i]} $floats $runs"
	echo "$(date) -- $version $workload ${BLOCKS[$i]} ${THREADS[$i]} $floats $runs" >> $DIRECTORY/log.txt
	res=$(timeout 15m ./a.out $floats ---gpumpi -g ${BLOCKS[$i]} -b ${THREADS[$i]} 2>> $DIRECTORY/log.txt)
	echo "$res" | while read line; do echo "$version $workload $runs ${BLOCKS[$i]} ${THREADS[$i]} $floats $line"; done >> $DIRECTORY/out.csv
done
done
done
done
done
