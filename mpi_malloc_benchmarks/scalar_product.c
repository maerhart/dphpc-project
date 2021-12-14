#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>

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
  double prod;
  int my_rank;
  int num_procs;
  double start;
  double t_malloc, t_work, t_free;

  if (argc != 2) return -1;
  int n_elems = atoi(argv[1]);
 
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  start = MPI_Wtime();
  double *local_x = (double *) malloc(n_elems * sizeof(double));
  double *local_y = (double *) malloc(n_elems * sizeof(double));
  t_malloc = MPI_Wtime() - start;

  start = MPI_Wtime();
  for (int j = 0; j < 500000; ++j) {
    for(i = 0; i < n_elems; i++) {
      local_x[i] = 0.01 * i;
      local_y[i] = 0.03 * i;
    }
    double local_prod;
    local_prod = dotProduct(local_x,local_y,n_elems);
  }
  t_work = MPI_Wtime() - start;
  // MPI_Reduce(&local_prod, &prod, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  start = MPI_Wtime();
  free(local_x);
  free(local_y);
  t_free = MPI_Wtime() - start;

  double malloc_sum, work_sum, free_sum;
  MPI_Reduce(&t_malloc, &malloc_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&t_work, &work_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&t_free, &free_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  if (my_rank == 0) {
    // printf("dotProduct = %f\n", prod);
    printf("%f\n", malloc_sum/num_procs);
    printf("%f\n", work_sum/num_procs);
    printf("%f\n", free_sum/num_procs);
  } 

  MPI_Finalize();

  return 0;
}
