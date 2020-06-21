
## Build

### With llvm and clang installed in system:

```
mkdir build-gpu_mpi
cd build-gpu_mpi
cmake ../gpu_mpi -GNinja
cmake --build .
```

### With self compiled llvm and clang

Compile and install llvm with clang in user directory `llvm-install`.

```
wget https://releases.llvm.org/8.0.0/llvm-8.0.0.src.tar.xz
wget https://releases.llvm.org/8.0.0/cfe-8.0.0.src.tar.xz
wget https://releases.llvm.org/8.0.0/openmp-8.0.0.src.tar.xz
tar xvJf llvm-8.0.0.src.tar.xz && rm -rf llvm && mv llvm-8.0.0.src llvm
tar xvJf cfe-8.0.0.src.tar.xz && rm -rf clang && mv cfe-8.0.0.src clang
tar xvJf openmp-8.0.0.src.tar.xz && rm -rf openmp && mv openmp-8.0.0.src openmp
mkdir build-llvm && cd build-llvm
cmake ../llvm -DLLVM_ENABLE_PROJECTS="clang;openmp" -DCMAKE_BUILD_TYPE=RelWithDebInfo -DBUILD_SHARED_LIBS=ON -DLLVM_TARGETS_TO_BUILD="" -DLLVM_ENABLE_RTTI=ON -DCMAKE_INSTALL_PREFIX=../llvm-install/ -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ -GNinja
cmake --build .
cmake --build . --target install
```

Build this project

```
mkdir build-gpu_mpi
cd build-gpu_mpi
cmake ../gpu_mpi -DLLVM_DIR=../llvm-install/lib/cmake/llvm -DClang_DIR=../llvm-install/lib/cmake/clang
cmake --build .
```

Run tests

Run `ctest` from build directory or invoke `cmake` to run all tests

```
cmake --build . --target test
```

To debug individual tests with cuda-gdb, invoke following command from the build dir

```
cuda-gdb --args tests/sample_test
```
