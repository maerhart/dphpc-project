mkdir build
mkdir results
rm results/malloc_non_coalesced.csv
rm results/malloc_coalesced.csv
nvcc malloc_source_comparison.cu -o build/out

for ints in {20..30}
do
	for i in {1..3}
	do
		nc=$(./build/out $ints 0 1)
		c=$(./build/out $ints 1 1)
		echo "$ints $nc" >> results/malloc_non_coalesced.csv
		echo "$ints $c" >> results/malloc_coalesced.csv
	done	
done
