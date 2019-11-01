#!/bin/bash

# should be executed from build directory corresponding to CMakeLists.txt
cmake . -DCMAKE_BUILD_TYPE=Release

make -j4

./cuda_p2p_warp > benchmark_warp.txt
./cuda_p2p_block > benchmark_block.txt
./cuda_p2p_device > benchmark_device.txt
./cuda_p2p_multi_device > benchmark_multi_device.txt
./cuda_p2p_host_device > benchmark_host_device.txt

