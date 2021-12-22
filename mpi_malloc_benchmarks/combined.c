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
  double t_malloc=0, t_work=0, t_free=0;

  if (argc != 2) return -1;
  int n_elems = atoi(argv[1]);
 
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  if (my_rank < num_procs/2) {
    start = MPI_Wtime();
    double *local_x = (double *) malloc(n_elems * sizeof(double));
    double *local_y = (double *) malloc(n_elems * sizeof(double));
    t_malloc = MPI_Wtime() - start;

    for(i = 0; i < n_elems; i++) {
      local_x[i] = 0.01 * i;
      local_y[i] = 0.03 * i;
    }

    double local_prod;
    // Warmup
    for (int j = 0; j < 3; ++j) {
      local_prod = dotProduct(local_x,local_y,n_elems);
    }
    start = MPI_Wtime();
    for (int j = 0; j < 500000; ++j) {
      local_prod = dotProduct(local_x,local_y,n_elems);
    }
    t_work = MPI_Wtime() - start;
    // MPI_Reduce(&local_prod, &prod, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    start = MPI_Wtime();
    free(local_x);
    free(local_y);
    t_free = MPI_Wtime() - start;
  }

  start = MPI_Wtime();
  double *local_x = (double *) malloc(n_elems * sizeof(double));
  double *local_y = (double *) malloc(n_elems * sizeof(double));
  t_malloc += MPI_Wtime() - start;

  for(i = 0; i < n_elems; i++) {
    local_x[i] = 0.01 * i;
    local_y[i] = 0.03 * i;
  }

  double local_prod;
  // Warmup
  for (int j = 0; j < 3; ++j) {
    local_prod = dotProduct(local_x,local_y,n_elems);
  }
  start = MPI_Wtime();
  for (int j = 0; j < 500000; ++j) {
    local_prod = dotProduct(local_x,local_y,n_elems);
  }
  t_work += MPI_Wtime() - start;
  // MPI_Reduce(&local_prod, &prod, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  start = MPI_Wtime();
  free(local_x);
  free(local_y);
  t_free += MPI_Wtime() - start;

  printf("%f %f %f\n", t_malloc, t_work, t_free);

  MPI_Finalize();

  return 0;
}
