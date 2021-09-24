#ifndef INTERPDENS_AUX_H
#define INTERPDENS_AUX_H

#include "Grid.h"

struct interpDens_aux {
  /** charged densities */
  FPinterp ***rhoh;  // rho hat defined at center cell
  /** J current densities */
  FPinterp ***Jxh;
  FPinterp ***Jyh;
  FPinterp ***Jzh;  // on nodes
};

/** allocated interpolated densities per species */
void interp_dens_aux_allocate(struct grid *grd, struct interpDens_aux *id_aux);

/** deallocate interpolated densities per species */
void interp_dens_aux_deallocate(struct grid *grd, struct interpDens_aux *id_aux);

#endif
