COMMIT=$(git log -n1 | head -n 1 | cut -c 8-)
DATE=$(date "+%F-%T")
SCRIPT="${BASH_SOURCE[0]:-$0}"
ROOTDIR="$(cd "$(dirname ${SCRIPT})/" && pwd)"
DIRECTORY=$ROOTDIR/results/$DATE #--$COMMIT
BUILD=$ROOTDIR/build
mkdir -p $DIRECTORY

HEADER="version workload runs warmup blocks threads_per_block num_floats malloc_mean malloc_max free_mean free_max work_mean work_max"
WORKLOADS="sum_reduce prod_reduce max_reduce pair_prod sum_all_prod"
VERSION="sum_reduce_baseline sum_reduce_v1 sum_reduce_v3"
WORKLOADS="sum_reduce"
BLOCKS=(192 96 48 24 12)
THREADS=(64 128 256 512 1024)

RUNS="5"
WU="0"

rm -r $BUILD
mkdir $BUILD
cp *.cu* $BUILD
cp build/benchmarks_template.cu $DIRECTORY
cp $BUILD/dynamic_allocator.cu $DIRECTORY
echo $HEADER >> $DIRECTORY/out.csv
for version in $VERSION
do
for workload in $WORKLOADS
do
	rm $BUILD/benchmarks_replaced.cu
	rm $BUILD/out
	sed -e "s/\${VERSION}/${version}/" -e "s/\${WORKLOAD}/${workload}/" $ROOTDIR/benchmarks_template.cu > $BUILD/benchmarks_replaced.cu
	nvcc $BUILD/benchmarks_replaced.cu -o $BUILD/out
for i in ${!BLOCKS[@]}
do
for floats in {1..20}
do
for runs in $RUNS
do
for wu in $WU
do
	cd $BUILD
	echo "${BLOCKS[$i]} ${THREADS[$i]} $runs $wu $floats"
	res=$(./out ${BLOCKS[$i]} ${THREADS[$i]} $runs $wu $floats)
	echo "$res" | while read line; do echo "$version $workload $runs $wu ${BLOCKS[$i]} ${THREADS[$i]} $floats $line"; done >> $DIRECTORY/out.csv
done
done
done
done
done
done
rm -r $BUILD