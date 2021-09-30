#!/bin/bash

scriptdir=$(readlink -f $(dirname "$0"))
projectdir="$scriptdir/../"
llvmcompiledir="$projectdir/llvm/"
llvmbuild="$llvmcompiledir/build-llvm/"
llvminstall="$llvmcompiledir/install-llvm/"
gpumpibuild="$projectdir/build/"

numthreads=$(grep -c ^processor /proc/cpuinfo)

export CMAKE_BUILD_PARALLEL_LEVEL=$numthreads

if [ ! -d "$llvmbuild" ]; then

mkdir -p "$llvmcompiledir"
cd "$llvmcompiledir"

wget --continue https://github.com/llvm/llvm-project/releases/download/llvmorg-11.0.1/llvm-11.0.1.src.tar.xz
wget --continue https://github.com/llvm/llvm-project/releases/download/llvmorg-11.0.1/clang-11.0.1.src.tar.xz
wget --continue https://github.com/llvm/llvm-project/releases/download/llvmorg-11.0.1/openmp-11.0.1.src.tar.xz

mkdir -p llvm && tar --extract --xz --file llvm-11.0.1.src.tar.xz --strip-components=1 --directory llvm
mkdir -p clang && tar --extract --xz --file clang-11.0.1.src.tar.xz --strip-components=1 --directory clang
mkdir -p openmp && tar --extract --xz --file openmp-11.0.1.src.tar.xz --strip-components=1 --directory openmp

mkdir "$llvmbuild"

fi

if [ ! -d "$llvminstall" ]; then

cd "$llvmbuild"
cmake "$llvmcompiledir/llvm" -DLLVM_ENABLE_PROJECTS="clang;openmp" -DCMAKE_BUILD_TYPE=RelWithDebInfo -DBUILD_SHARED_LIBS=ON -DLLVM_TARGETS_TO_BUILD="" -DLLVM_ENABLE_RTTI=ON -DCMAKE_INSTALL_PREFIX="$llvminstall" -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++
cmake --build .
cmake --build . --target install

fi

if [ ! -d "$gpumpibuild" ]; then

mkdir "$gpumpibuild"
cd "$gpumpibuild"
cmake "$projectdir" -DLLVM_DIR="$llvminstall/lib/cmake/llvm" -DClang_DIR="$llvminstall/lib/cmake/clang" 

fi

cmake --build "$gpumpibuild"