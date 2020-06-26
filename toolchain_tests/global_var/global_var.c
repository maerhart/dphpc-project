#include <stdio.h>

#include <mpi.h>

int g1 = -15;

int g2[2] = { -1, -2 };

int error = 0;

#define FAIL_IF(x) do { if (x) { printf("error %s:%d %s!\n", __FILE__, __LINE__, #x); error = 1; } } while (0)

int main() {
    MPI_Init(0, 0);

    int rank = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    FAIL_IF(g1 != -15);
    FAIL_IF(g2[0] != -1);
    FAIL_IF(g2[1] != -2);

    g1 = rank;
    g2[0] = rank + 1;
    g2[1] = rank + 2;

    MPI_Barrier(MPI_COMM_WORLD);

    FAIL_IF(g1 != rank);
    FAIL_IF(g2[0] != rank + 1);
    FAIL_IF(g2[1] != rank + 2);

    MPI_Finalize();

    return error;
}
