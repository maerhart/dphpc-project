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

  // init memory
  if (my_rank == 0) {
    for(i = 0; i < n_elems; i++) {
      local_x[i] = i;
    }
  }

  // Warmup
  MPI_Bcast(local_x, n_elems, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  // measure work to evaluate memory access time
  start = MPI_Wtime();
  MPI_Bcast(local_x, n_elems, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  t_work = MPI_Wtime() - start;

  // check that the result of the computations are indeed correct
  for (int i =0; i < n_elems; i++) {
    assert(local_x[i] == i);
    if (local_x[i] != i) {
      printf("Expected %f, but got %f\n", local_x[i], (double)i);
      fflush(stdout);
    }
  }

  // measure free time
  start = MPI_Wtime();
  free(local_x);
  t_free = MPI_Wtime() - start;

  printf("%f %f %f\n", t_malloc, t_work, t_free);

  MPI_Finalize();

  return 0;
}
