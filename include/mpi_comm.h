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

void mpi_reduce_dens_net(struct grid*, struct interpDensNet*);
void mpi_reduce_dens_spec(struct grid*, struct interpDensSpecies*);
void mpi_broadcast_field(struct grid *grd, struct EMfield *field);
void mpi_scatter_particles(struct particles *part_global, struct particles *part_local);

#endif