// RUN: convert.sh %s | sed '/^[[:blank:]]*[//]/d' | FileCheck %s

// CHECK-LABEL: __device__ void f(, bool __coalesced = false)
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

// CHECK-LABEL: void g(, bool __coalesced = false)
void g() {
    // CHECK-NEXT: void *ptr = malloc(sizeof(int), __coalesced);
    // CHECK-NEXT: free(ptr);
    // CHECK-NEXT: ptr = malloc(sizeof(int), __coalesced);
    // CHECK-NEXT: free(ptr);
    void *ptr = malloc(sizeof(int));
    free(ptr);
    ptr = malloc(sizeof(int));
    free(ptr);
}

// CHECK-NEXT: int __gpu_main(int argc, char** argv)
int main(int argc, char **argv) {
    // CHECK-NEXT: void *ptr = malloc(sizeof(int), true);
    // CHECK-NEXT: free(ptr);
    // CHECK-NEXT: f(, true);
    // CHECK-NEXT: if (argc == 2) {
    // CHECK-NEXT:     g();
    // CHECK-NEXT: }
    // CHECK-NEXT: return 0;
    void *ptr = malloc(sizeof(int));
    free(ptr);

    f();

    if (argc == 2) {
        g();
    }

    return 0;
}