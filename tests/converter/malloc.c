// RUN: convert.sh %s | sed '/^[[:blank:]]*[//]/d' | FileCheck %s
#include "mpi.h"
#include <stdlib.h>

// CHECK-LABEL: __device__ void f(bool __coalesced = false)
void f() {
    // CHECK-NEXT: void *ptr = malloc(sizeof(int), __coalesced);
    // CHECK-NEXT: free(ptr);
    // CHECK-NEXT: ptr = malloc(sizeof(int), __coalesced);
    // CHECK-NEXT: free(ptr);
    void *ptr = malloc(sizeof(int));
    free(ptr);
    ptr = malloc(sizeof(int));
    free(ptr);
}

// CHECK-LABEL: void g(bool __coalesced = false)
void g() {
    // CHECK-NEXT: void *ptr = malloc(sizeof(int), __coalesced);
    // CHECK-NEXT: free(ptr);
    // CHECK-NEXT: ptr = malloc(sizeof(int), __coalesced);
    // CHECK-NEXT: int recv;
    // CHECK-NEXT: MPI_Alltoall(ptr, 4, MPI_INT, (void *)&recv, 4, MPI_INT, MPI_COMM_WORLD, __coalesced);
    // CHECK-NEXT: free(ptr);
    void *ptr = malloc(sizeof(int));
    free(ptr);
    ptr = malloc(sizeof(int));
    int recv;
    MPI_Alltoall(ptr, 4, MPI_INT, &recv, 4, MPI_INT, MPI_COMM_WORLD);
    free(ptr);
}

// CHECK-LABEL: void h(void *ptr, bool __coalesced = false)
void h(void *ptr) {
    // CHECK-NEXT: int recv;
    // CHECK-NEXT: MPI_Alltoall(ptr, 4, MPI_INT, (void *)&recv, 4, MPI_INT, MPI_COMM_WORLD, __coalesced);
    int recv;
    MPI_Alltoall(ptr, 4, MPI_INT, &recv, 4, MPI_INT, MPI_COMM_WORLD);
}

// CHECK-LABEL: int __gpu_main(int argc, char** argv)
int main(int argc, char **argv) {
    // CHECK-NEXT: init_malloc();
    // CHECK-NEXT: void *ptr = malloc(sizeof(int), true);
    // CHECK-NEXT: f(true);
    // CHECK-NEXT: int recv;
    // CHECK-NEXT: MPI_Alltoall(ptr, 4, MPI_INT, (void *)&recv, 4, MPI_INT, MPI_COMM_WORLD, true);
    // CHECK-NEXT: free(ptr);
    // CHECK-NEXT: if (argc == 2) {
    // CHECK-NEXT:     g();
    // CHECK-NEXT: }
    // CHECK-NEXT: clean_malloc();
    // CHECK-NEXT: return 0;
    void *ptr = malloc(sizeof(int));
    f();
    int recv;
    MPI_Alltoall(ptr, 4, MPI_INT, &recv, 4, MPI_INT, MPI_COMM_WORLD);
    free(ptr);
    if (argc == 2) {
        g();
    }
    return 0;
}
