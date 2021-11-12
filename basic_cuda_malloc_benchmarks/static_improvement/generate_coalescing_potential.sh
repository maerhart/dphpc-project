COMMIT=$(git log -n1 | head -n 1 | cut -c 8-)
DATE=$(date "+%F-%T")
SCRIPT="${BASH_SOURCE[0]:-$0}"
ROOTDIR="$(cd "$(dirname ${SCRIPT})/" && pwd)"
DIRECTORY=$ROOTDIR/results/$DATE--$COMMIT
mkdir -p $DIRECTORY
mkdir build
nvcc malloc_source_comparison.cu -o build/out

for ints in {20..30}
do
	for threads in 0 7 9 10
	do
		for blocks in 0 4 6 10
		do
			for i in {1..10}
			do
				echo "$ints $i $threads $blocks"
				nc=$(./build/out $ints 0 $threads $blocks 1)
				c=$(./build/out $ints 1 $threads $blocks 1)
				echo "$ints $nc" >> $DIRECTORY/malloc_non_coalesced.csv
				echo "$ints $c" >> $DIRECTORY/malloc_coalesced.csv
			done
		done
	done	
done
