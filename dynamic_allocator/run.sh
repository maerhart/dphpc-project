COMMIT=$(git log -n1 | head -n 1 | cut -c 8-)
DATE=$(date "+%F-%T")
SCRIPT="${BASH_SOURCE[0]:-$0}"
ROOTDIR="$(cd "$(dirname ${SCRIPT})/" && pwd)"
DIRECTORY=$ROOTDIR/results/$DATE #--$COMMIT
BUILD=$ROOTDIR/build
mkdir -p $DIRECTORY
SOURCE=$ROOTDIR/../gpu_libs/gpu_malloc

HEADER="version workload runs warmup blocks threads_per_block num_floats malloc_mean malloc_max free_mean free_max work_mean work_max"
WORKLOADS="sum_reduce sum_all_prod" # dot_product pair_prod sum_all_prod" #max_reduce
VERSION="baseline v1_martin v3_nils v5_anton v6_nils" #baseline v1_martin v3_nils v5_anton v6_nils
BLOCKS=(12) #(192 96 48 24 12)
THREADS=(1024) #(64 128 256 512 1024)

FLOATS=()
#for i in {0..9} # powers of 2
#for i in {1..100} # linear
for i in {1..20} # every 25th
#for i in {6..7} # every 25th, 150, 175
#for i in {18..24} # every 25th, 450 to 600
do
    #FLOATS+=( $((2**$i)) ) # powers of 2
    #FLOATS+=( $i ) # linear
    FLOATS+=( $((25*$i)) ) # every 25th
done

RUNS="20"
WU="2"

rm -r $BUILD
mkdir $BUILD
cp *.cu* $BUILD
cp $SOURCE/*.cu* $BUILD
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
for floats in ${FLOATS[@]}
do
for runs in $RUNS
do
for wu in $WU
do
	cd $BUILD
	echo "$version $workload ${BLOCKS[$i]} ${THREADS[$i]} $runs $wu $floats"
	res=$(./out ${BLOCKS[$i]} ${THREADS[$i]} $runs $wu $floats)
	echo "$res" | while read line; do echo "$version $workload $runs $wu ${BLOCKS[$i]} ${THREADS[$i]} $floats $line"; done >> $DIRECTORY/out.csv
done
done
done
done
done
done
rm -r $BUILD
