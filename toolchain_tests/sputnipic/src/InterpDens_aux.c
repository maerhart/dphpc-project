#include "InterpDens_aux.h"

#include "Alloc.h"

/** allocated interpolated densities per species */
void interp_dens_aux_allocate(struct grid *grd,
                                     struct interpDens_aux *id_aux) {
  // hat - charge density defined on nodes and center cell
  id_aux->rhoh = (FPinterp***) newArr(sizeof(FPinterp), 3, grd->nxc, grd->nyc, grd->nzc);
  // hat - current
  id_aux->Jxh = (FPinterp***) newArr(sizeof(FPinterp), 3, grd->nxn, grd->nyn, grd->nzn);
  id_aux->Jyh = (FPinterp***) newArr(sizeof(FPinterp), 3, grd->nxn, grd->nyn, grd->nzn);
  id_aux->Jzh = (FPinterp***) newArr(sizeof(FPinterp), 3, grd->nxn, grd->nyn, grd->nzn);

}

/** deallocate interpolated densities per species */
void interp_dens_aux_deallocate(struct grid *grd,
                                       struct interpDens_aux *id_aux) {
  // hat - charge density
  delArr(3, id_aux->rhoh);
  // hat - current
  delArr(3, id_aux->Jxh);
  delArr(3, id_aux->Jyh);
  delArr(3, id_aux->Jzh);
}