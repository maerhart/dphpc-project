#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>


#define MPI_CHECK(e) do { if ((e) != MPI_SUCCESS) { printf("MPI_CHECK failed %s:%d %s\n", __FILE__, __LINE__, #e); exit(1); } } while (0)


void benchmark_point_to_point() {
    int size, rank;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &size));


    for (int enabled_pairs = 1; enabled_pairs <= size / 2; enabled_pairs++) {
        for (int buffer_size = 1; buffer_size <= 1<<20; buffer_size *= 2) {
            char* buffer = (char*) calloc(buffer_size, sizeof(char));

            double avg_time = 0;
            if (rank / 2 < enabled_pairs) {
                double total_time = 0;
                int repetitions = 0;
                do {
                    double start_time = MPI_Wtime();
                    if (rank % 2 == 0) {
                        if (total_time >= 0.1) buffer[0] = 1;
                        MPI_CHECK(MPI_Send(buffer, buffer_size, MPI_CHAR, rank + 1, 10, MPI_COMM_WORLD));
                        MPI_CHECK(MPI_Recv(buffer, buffer_size, MPI_CHAR, rank + 1, 10, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
                    } else {
                        MPI_CHECK(MPI_Recv(buffer, buffer_size, MPI_CHAR, rank - 1, 10, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
                        MPI_CHECK(MPI_Send(buffer, buffer_size, MPI_CHAR, rank - 1, 10, MPI_COMM_WORLD));
                    }
                    total_time += MPI_Wtime() - start_time;
                    repetitions++;
                } while (buffer[0] == 0);

                avg_time = total_time / repetitions;
            }

            // print result from slowest process
            double out_time = -1;
            MPI_CHECK(MPI_Reduce(&avg_time, &out_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));

            free(buffer);

            if (rank == 0) {
                printf("point_to_point enabled_pairs=%d buffer_size=%d time=%f us\n", enabled_pairs, buffer_size, out_time * 1e6);

            }
        }
    }


}

void benchmark_one_to_many() {
    int size, rank;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &size));

    for (int enabled_ranks = 1; enabled_ranks <= size; enabled_ranks++) {
        MPI_Group world_group;
        MPI_CHECK(MPI_Comm_group(MPI_COMM_WORLD, &world_group));
        int* included_ranks = (int*) malloc(enabled_ranks * sizeof(int));
        assert(included_ranks && "Can't allocate memory\n");
        for (int i = 0; i < enabled_ranks; i++) {
            included_ranks[i] =  i;
        }
        MPI_Group enabled_group;
        MPI_CHECK(MPI_Group_incl(world_group, enabled_ranks, included_ranks, &enabled_group));

        MPI_Comm enabled_comm;
        MPI_CHECK(MPI_Comm_create(MPI_COMM_WORLD, enabled_group, &enabled_comm));

        for (int buffer_size = 1; buffer_size <= 1<<20; buffer_size *= 2) {
            char* buffer = (char*) calloc(buffer_size, sizeof(char));
            assert(buffer && "Can't allocate buffer\n");

            double avg_time = 0;
            double total_time = 0;
            int repetitions = 0;

            int src_process = 0;

            if (rank < enabled_ranks) {
                do {
                    if (rank == src_process && total_time >= 0.1) buffer[0] = 1;
                    double start_time = MPI_Wtime();
                    MPI_CHECK(MPI_Bcast(buffer, buffer_size, MPI_CHAR, src_process, enabled_comm));
                    total_time += MPI_Wtime() - start_time;
                    repetitions++;
                    src_process = (src_process + 1) % enabled_ranks;
                } while (buffer[0] == 0);

                avg_time = total_time / repetitions;
            }

            free(buffer);

            // print result from slowest process
            double out_time = -1;
            MPI_CHECK(MPI_Reduce(&avg_time, &out_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));

            if (rank == 0) {
                printf("one_to_many enabled_ranks=%d buffer_size=%d time=%f us\n", enabled_ranks, buffer_size, out_time * 1e6);
            }
        }

        if (rank < enabled_ranks) {
            MPI_CHECK(MPI_Comm_free(&enabled_comm));
        }
        MPI_CHECK(MPI_Group_free(&enabled_group));

        free(included_ranks);

    }
}

void benchmark_many_to_one() {
    int size, rank;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &size));

    for (int enabled_ranks = 1; enabled_ranks <= size; enabled_ranks++) {
        MPI_Group world_group;
        MPI_CHECK(MPI_Comm_group(MPI_COMM_WORLD, &world_group));
        int* included_ranks = (int*) malloc(enabled_ranks * sizeof(int));
        assert(included_ranks && "Can't allocate memory\n");
        for (int i = 0; i < enabled_ranks; i++) {
            included_ranks[i] =  i;
        }
        MPI_Group enabled_group;
        MPI_CHECK(MPI_Group_incl(world_group, enabled_ranks, included_ranks, &enabled_group));

        MPI_Comm enabled_comm;
        MPI_CHECK(MPI_Comm_create(MPI_COMM_WORLD, enabled_group, &enabled_comm));

        for (int buffer_size = 1; buffer_size <= 1<<20; buffer_size *= 2) {
            char* buffer = (char*) calloc(buffer_size, sizeof(char));
            char* recv_buffer = (char*) calloc(buffer_size * enabled_ranks, sizeof(char));
            assert(buffer && "Can't allocate buffer\n");

            double avg_time = 0;
            double total_time = 0;
            int repetitions = 0;

            int dst_process = 0;
            int finished = 0;

            if (rank < enabled_ranks) {
                do {
                    if (finished > 0) {
                        finished++;
                        buffer[0] = 1;
                    } else if (rank == 0 && dst_process == 0 && total_time >= 0.1) {
                        finished = 1;
                    }
                    double start_time = MPI_Wtime();
                    MPI_CHECK(MPI_Gather(buffer, buffer_size, MPI_CHAR, recv_buffer, buffer_size, MPI_CHAR, dst_process, enabled_comm));
                    total_time += MPI_Wtime() - start_time;
                    if (recv_buffer[0] == 1 && finished == 0) {
                        finished = rank + 1;
                    }
                    repetitions++;
                    dst_process = (dst_process + 1) % enabled_ranks;
                } while (finished < enabled_ranks);

                avg_time = total_time / repetitions;
            }

            free(buffer);
            free(recv_buffer);

            // print result from slowest process
            double out_time = -1;
            MPI_CHECK(MPI_Reduce(&avg_time, &out_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));

            if (rank == 0) {
                printf("many_to_one enabled_ranks=%d buffer_size=%d time=%f us\n", enabled_ranks, buffer_size, out_time * 1e6);
            }
        }

        if (rank < enabled_ranks) {
            MPI_CHECK(MPI_Comm_free(&enabled_comm));
        }
        MPI_CHECK(MPI_Group_free(&enabled_group));

        free(included_ranks);

    }
}

void benchmark_all_to_all() {
    int size, rank;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &size));

    for (int enabled_ranks = 1; enabled_ranks <= size; enabled_ranks++) {
        MPI_Group world_group;
        MPI_CHECK(MPI_Comm_group(MPI_COMM_WORLD, &world_group));
        int* included_ranks = (int*) malloc(enabled_ranks * sizeof(int));
        assert(included_ranks && "Can't allocate memory\n");
        for (int i = 0; i < enabled_ranks; i++) {
            included_ranks[i] =  i;
        }
        MPI_Group enabled_group;
        MPI_CHECK(MPI_Group_incl(world_group, enabled_ranks, included_ranks, &enabled_group));

        MPI_Comm enabled_comm;
        MPI_CHECK(MPI_Comm_create(MPI_COMM_WORLD, enabled_group, &enabled_comm));

        for (int buffer_size = 1; buffer_size <= 1<<20; buffer_size *= 2) {
            char* send_buffer = (char*) calloc(buffer_size * enabled_ranks, sizeof(char));
            assert(send_buffer && "Can't allocate buffer\n");
            char* recv_buffer = (char*) calloc(buffer_size * enabled_ranks, sizeof(char));
            assert(recv_buffer && "Can't allocate buffer\n");

            double avg_time = 0;
            double total_time = 0;
            int repetitions = 0;

            if (rank < enabled_ranks) {
                do {
                    if (rank == 0 && total_time >= 0.1) {
                        for (int k = 0; k < enabled_ranks; k++) {
                            send_buffer[k * buffer_size] = 1;
                        }
                    }
                    double start_time = MPI_Wtime();
                    MPI_CHECK(MPI_Alltoall(send_buffer, buffer_size, MPI_CHAR, recv_buffer, buffer_size, MPI_CHAR, enabled_comm));
                    total_time += MPI_Wtime() - start_time;
                    repetitions++;
                } while (recv_buffer[0] == 0);

                avg_time = total_time / repetitions;
            }

            free(send_buffer);
            free(recv_buffer);

            // print result from slowest process
            double out_time = -1;
            MPI_CHECK(MPI_Reduce(&avg_time, &out_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));

            if (rank == 0) {
                printf("all_to_all enabled_ranks=%d buffer_size=%d time=%f us\n", enabled_ranks, buffer_size, out_time * 1e6);
            }
        }

        if (rank < enabled_ranks) {
            MPI_CHECK(MPI_Comm_free(&enabled_comm));
        }
        MPI_CHECK(MPI_Group_free(&enabled_group));

        free(included_ranks);

    }
       
}


int main(int argc, char** argv) {
    MPI_CHECK(MPI_Init(&argc, &argv));

    benchmark_point_to_point();
    benchmark_one_to_many();
    benchmark_many_to_one();
    benchmark_all_to_all();

    MPI_CHECK(MPI_Finalize());

    return 0;
}
