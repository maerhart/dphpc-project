COMMIT=$(git log -n1 | head -n 1 | cut -c 8-)
DATE=$(date "+%F-%T")
SCRIPT="${BASH_SOURCE[0]:-$0}"
ROOTDIR="$(cd "$(dirname ${SCRIPT})/" && pwd)"
DIRECTORY=$ROOTDIR/results/$DATE--$COMMIT
mkdir -p $DIRECTORY
mkdir build
nvcc dynamic_coalescing_potential.cu -o build/out

THREADS="0 7 9 10"
BLOCKS="0 4 6 10"
#THREADS="7"
#BLOCKS="4"

for ints in {10..30}
do
	for threads in $THREADS
	do
		for blocks in $BLOCKS
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
rm -r build
