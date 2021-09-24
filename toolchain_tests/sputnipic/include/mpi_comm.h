#ifndef MPI_COMM_H
#define MPI_COMM_H

#include <mpi.h>
#include <stdio.h>
#include "PrecisionTypes.h"
#include "Grid.h"
#include "InterpDensNet.h"
#include "EMfield.h"
#include "Particles.h"
#include "Parameters.h"


void send_particle_batch(
	struct particles *part_send, 
	struct particles *part_recv, 
	size_t send_offset, 
	size_t recv_offset, 
	size_t len, 
	int send_rank,
	int recv_rank, 
	int rank
	);
void mpi_reduce_dens_net(struct grid*, struct interpDensNet*);
void mpi_reduce_dens_spec(struct grid*, struct interpDensSpecies*);
void mpi_broadcast_field(struct grid *grd, struct EMfield *field);
int mpi_scatter_particles(struct particles *part_global, struct particles *part_local, int offset, int batchsize, int num_particles);
void mpi_gather_particles(struct particles *part_global, struct particles *part_local, int offset, int batchsize, int num_particles);
#endif