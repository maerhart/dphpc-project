#include "InterpDensSpecies.h"

/** allocated interpolated densities per species */
void interp_dens_species_allocate(struct grid *grd,
                                  struct interpDensSpecies *ids, int is) {
  // set species ID
  ids->species_ID = is;

  // allocate 3D arrays
  // rho: 1
  ids->rhon = (FPinterp***) ptrArr((void**) &ids->rhon_flat, sizeof(FPinterp), 3, grd->nxn, grd->nyn, grd->nzn);  // nodes
  ids->rhoc = (FPinterp***) ptrArr((void**) &ids->rhoc_flat, sizeof(FPinterp), 3, grd->nxc, grd->nyc, grd->nzc);  // center
  // Jx: 2
  ids->Jx = (FPinterp***) ptrArr((void**) &ids->Jx_flat, sizeof(FPinterp), 3, grd->nxn, grd->nyn, grd->nzn);  // nodes
  // Jy: 3
  ids->Jy = (FPinterp***) ptrArr((void**) &ids->Jy_flat, sizeof(FPinterp), 3, grd->nxn, grd->nyn, grd->nzn);  // nodes
  // Jz: 4
  ids->Jz = (FPinterp***) ptrArr((void**) &ids->Jz_flat, sizeof(FPinterp), 3, grd->nxn, grd->nyn, grd->nzn);  // nodes
  // Pxx: 5
  ids->pxx = (FPinterp***) ptrArr((void**) &ids->pxx_flat, sizeof(FPinterp), 3, grd->nxn, grd->nyn, grd->nzn);  // nodes
  // Pxy: 6
  ids->pxy = (FPinterp***) ptrArr((void**) &ids->pxy_flat, sizeof(FPinterp), 3, grd->nxn, grd->nyn, grd->nzn);  // nodes
  // Pxz: 7
  ids->pxz = (FPinterp***) ptrArr((void**) &ids->pxz_flat, sizeof(FPinterp), 3, grd->nxn, grd->nyn, grd->nzn);  // nodes
  // Pyy: 8
  ids->pyy = (FPinterp***) ptrArr((void**) &ids->pyy_flat, sizeof(FPinterp), 3, grd->nxn, grd->nyn, grd->nzn);  // nodes
  // Pyz: 9
  ids->pyz = (FPinterp***) ptrArr((void**) &ids->pyz_flat, sizeof(FPinterp), 3, grd->nxn, grd->nyn, grd->nzn);  // nodes
  // Pzz: 10
  ids->pzz = (FPinterp***) ptrArr((void**) &ids->pzz_flat, sizeof(FPinterp), 3, grd->nxn, grd->nyn, grd->nzn);  // nodes
}

/** deallocate interpolated densities per species */
void interp_dens_species_deallocate(struct grid *grd,
                                    struct interpDensSpecies *ids) {
  // deallocate 3D arrays
  delArr(3, ids->rhon);
  delArr(3, ids->rhoc);
  // deallocate 3D arrays: J - current
  delArr(3, ids->Jx);
  delArr(3, ids->Jy);
  delArr(3, ids->Jz);
  // deallocate 3D arrays: pressure
  delArr(3, ids->pxx);
  delArr(3, ids->pxy);
  delArr(3, ids->pxz);
  delArr(3, ids->pyy);
  delArr(3, ids->pyz);
  delArr(3, ids->pzz);
}

/** deallocate interpolated densities per species */
void interpN2Crho(struct interpDensSpecies *ids, struct grid *grd) {
#pragma omp parallel for
  for (int i = 1; i < grd->nxc - 1; i++)
    for (int j = 1; j < grd->nyc - 1; j++)
#pragma clang loop vectorize(enable)
      for (int k = 1; k < grd->nzc - 1; k++) {
        ids->rhoc[i][j][k] =
            .125 *
            (ids->rhon[i][j][k] + ids->rhon[i + 1][j][k] +
             ids->rhon[i][j + 1][k] + ids->rhon[i][j][k + 1] +
             ids->rhon[i + 1][j + 1][k] + ids->rhon[i + 1][j][k + 1] +
             ids->rhon[i][j + 1][k + 1] + ids->rhon[i + 1][j + 1][k + 1]);
      }
}
