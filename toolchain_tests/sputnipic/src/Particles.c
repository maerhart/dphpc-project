#include "Particles.h"
#include <math.h>
#include "Alloc.h"

/** allocate particle arrays */
void particle_allocate(struct parameters *param, struct particles *part,
                       int is) {
  // set species ID
  part->species_ID = is;
  // number of particles
  part->nop = param->np[is];
  // maximum number of particles
  part->npmax = param->npMax[is];

  // choose a different number of mover iterations for ions and electrons
  part->NiterMover = param->NiterMover;
  part->n_sub_cycles = param->n_sub_cycles;

  // particles per cell
  part->npcelx = param->npcelx[is];
  part->npcely = param->npcely[is];
  part->npcelz = param->npcelz[is];
  part->npcel = part->npcelx * part->npcely * part->npcelz;

  // cast it to required precision
  part->qom = (FPpart)param->qom[is];

  long npmax = part->npmax;

  // initialize drift and thermal velocities
  // drift
  part->u0 = (FPpart)param->u0[is];
  part->v0 = (FPpart)param->v0[is];
  part->w0 = (FPpart)param->w0[is];
  // thermal
  part->uth = (FPpart)param->uth[is];
  part->vth = (FPpart)param->vth[is];
  part->wth = (FPpart)param->wth[is];

  //////////////////////////////
  /// ALLOCATION PARTICLE ARRAYS
  //////////////////////////////
  part->x = (FPpart*) newArr(sizeof(FPpart), 1, npmax); // new FPpart[npmax];
  part->y = (FPpart*) newArr(sizeof(FPpart), 1, npmax); // new FPpart[npmax];
  part->z = (FPpart*) newArr(sizeof(FPpart), 1, npmax); // new FPpart[npmax];
  // allocate velocity
  part->u = (FPpart*) newArr(sizeof(FPpart), 1, npmax); // new FPpart[npmax];
  part->v = (FPpart*) newArr(sizeof(FPpart), 1, npmax); // new FPpart[npmax];
  part->w = (FPpart*) newArr(sizeof(FPpart), 1, npmax); // new FPpart[npmax];
  // allocate charge = q * statistical weight
  part->q = (FPinterp*) newArr(sizeof(FPinterp), 1, npmax); // new FPinterp[npmax];

  part->track_particle = (bool*) newArr(sizeof(bool), 1, npmax); // new bool[npmax];
}
/** deallocate */
void particle_deallocate(struct particles *part) {
  // deallocate particle variables
  delArr(1, part->x); // delete[] part->x;
  delArr(1, part->y); // delete[] part->y;
  delArr(1, part->z); // delete[] part->z;
  delArr(1, part->u); // delete[] part->u;
  delArr(1, part->v); // delete[] part->v;
  delArr(1, part->w); // delete[] part->w;
  delArr(1, part->q); // delete[] part->q;
  delArr(1, part->track_particle); // delete[] part->track_particle;
}

/** allocate particle arrays */
void particle_aux_allocate(struct particles *part,
                           struct particles_aux *part_aux, int is) {
  // set species ID
  part_aux->species_ID = is;
  // number of particles
  part_aux->nop = part->nop;
  // maximum number of particles
  part_aux->npmax = part->npmax;

  // allocate densities brought by each particle
  part_aux->rho_p = (FPpart****) newArr(sizeof(FPpart), 4, part->npmax, 2, 2, 2); // new FPpart[part->npmax][2][2][2];
  part_aux->Jx = (FPpart****) newArr(sizeof(FPpart), 4, part->npmax, 2, 2, 2); // new FPpart[part->npmax][2][2][2];
  part_aux->Jy = (FPpart****) newArr(sizeof(FPpart), 4, part->npmax, 2, 2, 2); // new FPpart[part->npmax][2][2][2];
  part_aux->Jz = (FPpart****) newArr(sizeof(FPpart), 4, part->npmax, 2, 2, 2); // new FPpart[part->npmax][2][2][2];
  part_aux->pxx = (FPpart****) newArr(sizeof(FPpart), 4, part->npmax, 2, 2, 2); // new FPpart[part->npmax][2][2][2];
  part_aux->pxy = (FPpart****) newArr(sizeof(FPpart), 4, part->npmax, 2, 2, 2); // new FPpart[part->npmax][2][2][2];
  part_aux->pxz = (FPpart****) newArr(sizeof(FPpart), 4, part->npmax, 2, 2, 2); // new FPpart[part->npmax][2][2][2];
  part_aux->pyy = (FPpart****) newArr(sizeof(FPpart), 4, part->npmax, 2, 2, 2); // new FPpart[part->npmax][2][2][2];
  part_aux->pyz = (FPpart****) newArr(sizeof(FPpart), 4, part->npmax, 2, 2, 2); // new FPpart[part->npmax][2][2][2];
  part_aux->pzz = (FPpart****) newArr(sizeof(FPpart), 4, part->npmax, 2, 2, 2); // new FPpart[part->npmax][2][2][2];

  // cell index
  part_aux->ix_p = (int*) newArr(sizeof(int), 1, part->npmax); // new int[part->npmax];
  part_aux->iy_p = (int*) newArr(sizeof(int), 1, part->npmax); // new int[part->npmax];
  part_aux->iz_p = (int*) newArr(sizeof(int), 1, part->npmax); // new int[part->npmax];
}

void particle_aux_deallocate(struct particles_aux *part_aux) {
  // deallocate auxiliary particle variables needed for particle interpolation
  delArr(4, part_aux->rho_p); // delete[] part_aux->rho_p;
  delArr(4, part_aux->Jx); // delete[] part_aux->Jx;
  delArr(4, part_aux->Jy); // delete[] part_aux->Jy;
  delArr(4, part_aux->Jz); // delete[] part_aux->Jz;
  delArr(4, part_aux->pxx); // delete[] part_aux->pxx;
  delArr(4, part_aux->pxy); // delete[] part_aux->pxy;
  delArr(4, part_aux->pxz); // delete[] part_aux->pxz;
  delArr(4, part_aux->pyy); // delete[] part_aux->pyy;
  delArr(4, part_aux->pyz); // delete[] part_aux->pyz;
  delArr(4, part_aux->pzz); // delete[] part_aux->pzz;

  delArr(1, part_aux->ix_p);
  delArr(1, part_aux->iy_p);
  delArr(1, part_aux->iz_p);
}
