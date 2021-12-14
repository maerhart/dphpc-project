#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

double dotProduct(double *x, double *y, int n) {
  int i;
  double prod = 0.0;
  for (i = 0; i < n; i++) {
    prod += x[i]*y[i];
  }
  return prod;
}

int main(int argc, char *argv[]) {
  int i;
  double prod_x, prod_y;
  int my_rank;
  int num_procs;
  double start;
  double t_malloc, t_work, t_free;

  if (argc != 2) return -1;
  int n_elems = atoi(argv[1]);
 
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  // measure malloc time
  start = MPI_Wtime();
  double *local_x = (double *) malloc(n_elems * sizeof(double));
  t_malloc = MPI_Wtime() - start;

  // init memory
  if (my_rank == 0) {
    for(i = 0; i < n_elems; i++) {
      local_x[i] = i;
    }
  }

  // measure work to evaluate memory access time
  start = MPI_Wtime();
  MPI_Bcast(local_x, n_elems, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  t_work = MPI_Wtime() - start;

  // check that the result of the computations are indeed correct
  for (int i =0; i < n_elems; i++) {
    assert(local_x[i] == i);
    if (local_x[i] != i) {
      printf("Expected %f, but got %f\n", local_x[i], (double)i);
    }
  }

  // measure free time
  start = MPI_Wtime();
  free(local_x);
  t_free = MPI_Wtime() - start;

  // Calculate the average of the time measurements of all threads
  double malloc_sum, work_sum, free_sum;
  MPI_Reduce(&t_malloc, &malloc_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&t_work, &work_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&t_free, &free_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  if (my_rank == 0) {
    printf("%f\n", malloc_sum/num_procs);
    printf("%f\n", work_sum/num_procs);
    printf("%f\n", free_sum/num_procs);
  } 

  MPI_Finalize();

  return 0;
}
