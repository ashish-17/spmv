#!/bin/bash

set -e

make clean all
rm -rf stats
mkdir stats

for filename in matrices/*.mtx; do
	./spmv -mat matrices/$(basename "$filename" .mtx).mtx -ivec matrices/$(basename "$filename" .mtx).vec -alg atomic -blockSize 32 -blockNum 512 >> stats/32_atomic.stats
	./spmv -mat matrices/$(basename "$filename" .mtx).mtx -ivec matrices/$(basename "$filename" .mtx).vec -alg atomic -blockSize 64 -blockNum 512 >> stats/64_atomic.stats
	./spmv -mat matrices/$(basename "$filename" .mtx).mtx -ivec matrices/$(basename "$filename" .mtx).vec -alg atomic -blockSize 128 -blockNum 512 >> stats/128_at0mic.stats
	./spmv -mat matrices/$(basename "$filename" .mtx).mtx -ivec matrices/$(basename "$filename" .mtx).vec -alg atomic -blockSize 256 -blockNum 256 >> stats/256_atomic.stats
	./spmv -mat matrices/$(basename "$filename" .mtx).mtx -ivec matrices/$(basename "$filename" .mtx).vec -alg segment -blockSize 32 -blockNum 512 >> stats/32_segment.stats
	./spmv -mat matrices/$(basename "$filename" .mtx).mtx -ivec matrices/$(basename "$filename" .mtx).vec -alg segment -blockSize 64 -blockNum 512 >> stats/64_segment.stats
	./spmv -mat matrices/$(basename "$filename" .mtx).mtx -ivec matrices/$(basename "$filename" .mtx).vec -alg segment -blockSize 128 -blockNum 512 >> stats/128_segment.stats
	./spmv -mat matrices/$(basename "$filename" .mtx).mtx -ivec matrices/$(basename "$filename" .mtx).vec -alg segment -blockSize 256 -blockNum 256 >> stats/256_segment.stats
	./spmv -mat matrices/$(basename "$filename" .mtx).mtx -ivec matrices/$(basename "$filename" .mtx).vec -alg design -blockSize 32 -blockNum 512 >> stats/32_design.stats
	./spmv -mat matrices/$(basename "$filename" .mtx).mtx -ivec matrices/$(basename "$filename" .mtx).vec -alg design -blockSize 64 -blockNum 512 >> stats/64_design.stats
	./spmv -mat matrices/$(basename "$filename" .mtx).mtx -ivec matrices/$(basename "$filename" .mtx).vec -alg design -blockSize 128 -blockNum 512 >> stats/128_design.stats
	./spmv -mat matrices/$(basename "$filename" .mtx).mtx -ivec matrices/$(basename "$filename" .mtx).vec -alg design -blockSize 256 -blockNum 256 >> stats/256_design.stats
done

