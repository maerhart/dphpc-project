
## Build

### With llvm and clang installed in system:

```
mkdir build-gpu_mpi
cd build-gpu_mpi
cmake ../gpu_mpi -GNinja
cmake --build .
```

### With self compiled llvm and clang

Compile and install llvm with clang in user directory `install-llvm`.

```
wget https://github.com/llvm/llvm-project/releases/download/llvmorg-11.0.1/llvm-11.0.1.src.tar.xz
wget https://github.com/llvm/llvm-project/releases/download/llvmorg-11.0.1/clang-11.0.1.src.tar.xz
wget https://github.com/llvm/llvm-project/releases/download/llvmorg-11.0.1/openmp-11.0.1.src.tar.xz
tar xvJf llvm-11.0.1.src.tar.xz && rm -rf llvm && mv llvm-11.0.1.src llvm
tar xvJf clang-11.0.1.src.tar.xz && rm -rf clang && mv clang-11.0.1.src clang
tar xvJf openmp-11.0.1.src.tar.xz && rm -rf openmp && mv openmp-11.0.1.src openmp
mkdir build-llvm && cd build-llvm
cmake ../llvm -DLLVM_ENABLE_PROJECTS="clang;openmp" -DCMAKE_BUILD_TYPE=RelWithDebInfo -DBUILD_SHARED_LIBS=ON -DLLVM_TARGETS_TO_BUILD="" -DLLVM_ENABLE_RTTI=ON -DCMAKE_INSTALL_PREFIX=../install-llvm/ -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ -GNinja
cmake --build .
cmake --build . --target install
```

Build this project

```
mkdir build-gpu_mpi
cd build-gpu_mpi
cmake ../gpu_mpi -DLLVM_DIR=../install-llvm/lib/cmake/llvm -DClang_DIR=../install-llvm/lib/cmake/clang
cmake --build .
```

Run tests

Run `ctest` from build directory or invoke `cmake` to run all tests

```
cmake --build . --target test
```

To list available test names run `ctest -N`. To run particular test (for example `sample_test`) run `ctest -R "^sample_test$"`.

To debug individual tests with cuda-gdb, invoke following command from the build dir

```
cuda-gdb --args tests/sample_test
```
