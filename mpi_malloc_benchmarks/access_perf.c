#include "mpi.h"
#include <stdio.h>


int main(int argc, char *argv[])
{
    int myid, numprocs, i;
    int res1, res2;
    double startwtime = 0.0, endwtime;
    int namelen;
    char processor_name[MPI_MAX_PROCESSOR_NAME];

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Get_processor_name(processor_name, &namelen);

    fprintf(stdout, "Process %d of %d is on %s\n", myid, numprocs, processor_name);
    fflush(stdout);

    int *ptr = (int*) malloc(sizeof(int));
    int *ptr2 = (int*) malloc(sizeof(int));

    if (myid == 0)
        startwtime = MPI_Wtime();

    *ptr=myid;
    *ptr2=myid*2;

    MPI_Reduce(ptr, &res1, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(ptr2, &res2, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (myid == 0) {
        endwtime = MPI_Wtime();
        printf("Result 1: %d, Result 2: %d\n", res1, res2);
        printf("wall clock time = %f\n", endwtime - startwtime);
        fflush(stdout);
    }

    free(ptr);
    free(ptr2);

    MPI_Finalize();
    return 0;
}
