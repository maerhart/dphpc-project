#include "mpi.h"
#include <stdio.h>


int main(int argc, char *argv[])
{
    int myid, numprocs, i;
    double startwtime[MPI_MAX_PROCESSOR_NAME], endwtime[MPI_MAX_PROCESSOR_NAME];
    double walltime[MPI_MAX_PROCESSOR_NAME];
    double maxtime;
    int namelen;
    char processor_name[MPI_MAX_PROCESSOR_NAME];

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Get_processor_name(processor_name, &namelen);

    fprintf(stdout, "Process %d of %d is on %s\n", myid, numprocs, processor_name);
    fflush(stdout);

    startwtime[myid] = MPI_Wtime();
    int *ptr = (int*) malloc(sizeof(int));
    int *ptr2 = (int*) malloc(sizeof(int));
    *ptr=myid;
    *ptr2=myid*2;
    free(ptr);
    free(ptr2);
    endwtime[myid] = MPI_Wtime();
    walltime[myid] = endwtime[myid] - startwtime[myid];

    MPI_Reduce(walltime, &maxtime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (myid == 0) {
        printf("wall clock time = %f\n", maxtime);
        fflush(stdout);
    }

    MPI_Finalize();
    return 0;
}
