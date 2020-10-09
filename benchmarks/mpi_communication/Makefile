
all: mpi_communication.c
	mpicc -O3 -o mpi_communication mpi_communication.c

debug: mpi_communication.c
	mpicc -g -o mpi_communication mpi_communication.c

.PHONY: clean

clean:
	rm -f mpi_communication
