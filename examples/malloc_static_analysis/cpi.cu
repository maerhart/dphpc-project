#include "global_vars.cuh"
/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 *  (C) 2001 by Argonne National Laboratory.
 *      See COPYRIGHT in top-level directory.
 */

#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

__device__ double f(double);


__device__ double g(double a, bool __coalesced = false)
{
    double *mem = (double*)dyn_malloc(sizeof(double), __coalesced);
    double *mem2 = (double*)dyn_malloc(sizeof(double), __coalesced);
    *mem = (4.0 / (1.0 + a * a));
    double res = *mem;
    free((void *)mem);
    free((void *)mem2);
    return res;
}

__device__ double f(double a, bool __coalesced = false)
{
    double *mem = (double*)dyn_malloc(sizeof(double), __coalesced);
    double *mem2 = (double*)dyn_malloc(sizeof(double), __coalesced);
    *mem = (4.0 / (1.0 + a * a));
    g(a, __coalesced);
    if (a > 0) {
        g(a);
    }
    double res = *mem;
    free((void *)mem);
    free((void *)mem2);
    return res;
}

__device__ int __gpu_main(int argc, char** argv)
{
    int n, myid, numprocs, i;
    double PI25DT = 3.141592653589793238462643;
    double mypi, pi, h, sum, x;
    double startwtime = 0.0, endwtime;
    int namelen;
    char processor_name[MPI_MAX_PROCESSOR_NAME];

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Get_processor_name(processor_name, &namelen);

    fprintf(stdout, "Process %d of %d is on %s\n", myid, numprocs, processor_name);
    fflush(stdout);

    n = 10000;  /* default # of rectangles */
    if (myid == 0)
        startwtime = MPI_Wtime();

    void *ptr = dyn_malloc(8, true);
    void *ptr2 = dyn_malloc(8, true);

    MPI_Bcast((void *)&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    free(ptr);
    free(ptr2);

    f(x, true);

    h = 1.0 / (double) n;
    sum = 0.0;
    /* A slightly better approach starts from large i and works back */
    for (i = myid + 1; i <= n; i += numprocs) {
        x = h * ((double) i - 0.5);
        sum += f(x);
    }
    mypi = h * sum;

    MPI_Reduce((const void *)&mypi, (void *)&pi, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (myid == 0) {
        endwtime = MPI_Wtime();
        printf("pi is approximately %.16f, Error is %.16f\n", pi, fabs(pi - PI25DT));
        printf("wall clock time = %f\n", endwtime - startwtime);
        fflush(stdout);
    }

    MPI_Finalize();
    return 0;
}
__device__ int __gpu_max_threads() {
    return 1024;
}

