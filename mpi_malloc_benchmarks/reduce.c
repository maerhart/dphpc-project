#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

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

  double *local_y;
  if (my_rank == 0) local_y = (double *) malloc(n_elems * sizeof(double));

  // init memory
  for(i = 0; i < n_elems; i++) {
    local_x[i] = i;
  }

  // Warmup
  MPI_Reduce(local_x, local_y, n_elems, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  // measure work to evaluate memory access time
  start = MPI_Wtime();
  MPI_Reduce(local_x, local_y, n_elems, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  t_work = MPI_Wtime() - start;

  // check that the result of the computations are indeed correct
  if (my_rank == 0) {
    for (int i =0; i < n_elems; i++) {
      assert(local_y[i] == i*num_procs);
      if (local_y[i] != i*num_procs) {
        printf("Expected %f, but got %f\n", local_y[i], (double)i*num_procs);
        fflush(stdout);
      }
    }
  }

  // measure free time
  start = MPI_Wtime();
  free(local_x);
  t_free = MPI_Wtime() - start;

  if (my_rank == 0) free(local_y);

  if (my_rank == 0) printf("malloc ");
  MPI_Barrier(MPI_COMM_WORLD);
  printf("%f ", t_malloc);
  MPI_Barrier(MPI_COMM_WORLD);
  if (my_rank == 0) printf("\n work ");
  MPI_Barrier(MPI_COMM_WORLD);
  printf("%f ", t_work);
  MPI_Barrier(MPI_COMM_WORLD);
  if (my_rank == 0) printf("\n free ");
  MPI_Barrier(MPI_COMM_WORLD);
  printf("%f ", t_free);
  MPI_Barrier(MPI_COMM_WORLD);
  if (my_rank == 0) printf("\n");

  MPI_Finalize();

  return 0;
}
