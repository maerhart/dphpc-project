#ifndef MOVER_H
#define MOVER_H

#include "EMfield.h"
#include "Grid.h"
#include "InterpDensSpecies.h"
#include "Parameters.h"
#include "Particles.h"


/** Move and interpolate particles on mpi worker threads */
int batch_update_particles(
	struct particles *part_global, 
	struct particles *part_local,
	struct EMfield *field,
	struct interpDensSpecies *ids,
	struct grid *grd,
	struct parameters *param,
	int batchsize,
	long tot_num_particles
	);

/** particle mover  using predictor-corrector */
int mover_PC(struct particles *part, struct EMfield *field, struct grid *grd,
             struct parameters *param);

/** particle mover  using predictor-corrector (formulated to be automatically
 * vectorized  */
int mover_PC_V(struct particles *part, struct EMfield *field, struct grid *grd,
               struct parameters *param);

/** Interpolation Particle --> Grid: This is for species */
void interpP2G(struct particles *part, struct interpDensSpecies *ids,
               struct grid *grd);

/** particle mover  + interpolation*/
int mover_interp(struct particles *part, struct EMfield *field,
                 struct interpDensSpecies *ids, struct grid *grd,
                 struct parameters *param);

#endif
