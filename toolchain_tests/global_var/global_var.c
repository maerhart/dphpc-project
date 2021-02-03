#include <stdio.h>

#include <mpi.h>

extern int g1;
extern int g1;
extern int g1;
extern int g1;
extern int g1;
int g1 = -15;

int g2[2] = { -1, -2 };

int *g3[3] = {NULL, NULL, NULL}, g4[2][3] = {{1,2,3},{4,5,6}};

#define FAIL_IF(x) do { if (x) { printf("error %s:%d %s!\n", __FILE__, __LINE__, #x); error = 1; } } while (0)

#define MACRO_g2 g2 

int foo() {
    static int x[10] = {4,5,6,7};
    x[3] += 1;
    return x[3];
}

int main() {
    MPI_Init(0, 0);

    static int g5[10] = {1,2};

    int rank = -1;
    int error = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    FAIL_IF(g1 != -15);
    FAIL_IF(g2[0] != -1);
    FAIL_IF(g2[1] != -2);

    g1 = rank;
    MACRO_g2[0] = rank + 1;
    MACRO_g2[1] = rank + 2;

    g3[2] = ((int*)NULL) + rank + 3;
    g4[1][1] = rank + 4;

    foo();
    foo();

    MPI_Barrier(MPI_COMM_WORLD);

    int foo_res = foo();

    FAIL_IF(g1 != rank);
    FAIL_IF(g2[0] != rank + 1);
    FAIL_IF(g2[1] != rank + 2);
    FAIL_IF(g3[2] != ((int*)NULL) + rank + 3);
    FAIL_IF(g4[1][1] != rank + 4);
    FAIL_IF(g4[1][2] != 6);
    FAIL_IF(g5[1] != 2);
    FAIL_IF(foo_res != 10);


    MPI_Finalize();

    return error;
}
