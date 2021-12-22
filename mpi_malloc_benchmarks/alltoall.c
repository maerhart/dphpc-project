#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>

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
  double *local_x = (double *) malloc(n_elems * num_procs * sizeof(double));
  double *local_y = (double *) malloc(n_elems * num_procs * sizeof(double));
  t_malloc = MPI_Wtime() - start;

  for(i = 0; i < num_procs; i++) {
    for(int j = 0; j < n_elems; j++) {
      local_x[i*n_elems+j] = i+0.1*j;
    }
  }
  // printf("[before] proc: %d -> [%f, %f], [%f, %f], [%f, %f], [%f, %f]\n", my_rank, local_x[0], local_x[1], local_x[2], local_x[3], local_x[4], local_x[5], local_x[6], local_x[7]);
  start = MPI_Wtime();
  MPI_Alltoall(local_x, n_elems, MPI_DOUBLE, local_y, n_elems, MPI_DOUBLE, MPI_COMM_WORLD);
  t_work = MPI_Wtime() - start;
  // printf("[after] proc: %d -> [%f, %f], [%f, %f], [%f, %f], [%f, %f]\n", my_rank, local_y[0], local_y[1], local_y[2], local_y[3], local_y[4], local_y[5], local_y[6], local_y[7]);

  start = MPI_Wtime();
  free(local_x);
  free(local_y);
  t_free = MPI_Wtime() - start;

  MPI_Barrier(MPI_COMM_WORLD);

  printf("%f %f %f\n", t_malloc, t_work, t_free);

  MPI_Finalize();

  return 0;
}
