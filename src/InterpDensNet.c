#include "InterpDensNet.h"

/** allocated interpolated densities per species */
void interp_dens_net_allocate(struct grid *grd, struct interpDensNet *idn) {
  // charge density defined on nodes and center cell
  idn->rhon = (FPinterp***) ptrArr((void**) &idn->rhon_flat, sizeof(FPinterp), 3, grd->nxn, grd->nyn, grd->nzn);
  idn->rhoc = (FPinterp***) ptrArr((void**) &idn->rhoc_flat, sizeof(FPinterp), 3, grd->nxc, grd->nyc, grd->nzc);
  // current
  idn->Jx = (FPinterp***) ptrArr((void**) &idn->Jx_flat, sizeof(FPinterp), 3, grd->nxn, grd->nyn, grd->nzn);
  idn->Jy = (FPinterp***) ptrArr((void**) &idn->Jy_flat, sizeof(FPinterp), 3, grd->nxn, grd->nyn, grd->nzn);
  idn->Jz = (FPinterp***) ptrArr((void**) &idn->Jz_flat, sizeof(FPinterp), 3, grd->nxn, grd->nyn, grd->nzn);
  // pressure tensor
  idn->pxx = (FPinterp***) ptrArr((void**) &idn->pxx_flat, sizeof(FPinterp), 3, grd->nxn, grd->nyn, grd->nzn);
  idn->pxy = (FPinterp***) ptrArr((void**) &idn->pxy_flat, sizeof(FPinterp), 3, grd->nxn, grd->nyn, grd->nzn);
  idn->pxz = (FPinterp***) ptrArr((void**) &idn->pxz_flat, sizeof(FPinterp), 3, grd->nxn, grd->nyn, grd->nzn);
  idn->pyy = (FPinterp***) ptrArr((void**) &idn->pyy_flat, sizeof(FPinterp), 3, grd->nxn, grd->nyn, grd->nzn);
  idn->pyz = (FPinterp***) ptrArr((void**) &idn->pyz_flat, sizeof(FPinterp), 3, grd->nxn, grd->nyn, grd->nzn);
  idn->pzz = (FPinterp***) ptrArr((void**) &idn->pzz_flat, sizeof(FPinterp), 3, grd->nxn, grd->nyn, grd->nzn);
}

/** deallocate interpolated densities per species */
void interp_dens_net_deallocate(struct grid *grd, struct interpDensNet *idn) {
  // charge density
  delArr(3, idn->rhon);
  delArr(3, idn->rhoc);
  // current
  delArr(3, idn->Jx);
  delArr(3, idn->Jy);
  delArr(3, idn->Jz);
  // pressure
  delArr(3, idn->pxx);
  delArr(3, idn->pxy);
  delArr(3, idn->pxz);
  delArr(3, idn->pyy);
  delArr(3, idn->pyz);
  delArr(3, idn->pzz);
}

/* set species densities to zero */
void setZeroSpeciesDensities(struct interpDensSpecies *ids, struct grid *grd,
                             int ns) {
  //////////////////////////////////
  // Densities per species
  for (int is = 0; is < ns; is++)
#pragma omp parallel for
    for (int i = 0; i < grd->nxn; i++)
      for (int j = 0; j < grd->nyn; j++)
        for (int k = 0; k < grd->nzn; k++) {
          // charge density
          ids[is].rhon[i][j][k] = 0.0;  // quantities defined on node
          // current
          ids[is].Jx[i][j][k] = 0.0;  // quantities defined on node
          ids[is].Jy[i][j][k] = 0.0;  // quantities defined on node
          ids[is].Jz[i][j][k] = 0.0;  // quantities defined on node
          // pressure
          ids[is].pxx[i][j][k] = 0.0;  // quantities defined on node
          ids[is].pxy[i][j][k] = 0.0;  // quantities defined on node
          ids[is].pxz[i][j][k] = 0.0;  // quantities defined on node
          ids[is].pyy[i][j][k] = 0.0;  // quantities defined on node
          ids[is].pyz[i][j][k] = 0.0;  // quantities defined on node
          ids[is].pzz[i][j][k] = 0.0;  // quantities defined on node
        }

  //////////////////////////////////
  //  rhoc  - center cell
  for (int is = 0; is < ns; is++)
#pragma omp parallel for
    for (int i = 0; i < grd->nxc; i++)
      for (int j = 0; j < grd->nyc; j++)
        for (int k = 0; k < grd->nzc; k++) {
          ids[is].rhoc[i][j][k] = 0.0;
        }
}

/* set species densities to zero */
void setZeroNetDensities(struct interpDensNet *idn, struct grid *grd) {
//////////////////////////////////////
// Net densities
// calculate the coordinates - Nodes
#pragma omp parallel for
  for (int i = 0; i < grd->nxn; i++)
    for (int j = 0; j < grd->nyn; j++)
      for (int k = 0; k < grd->nzn; k++) {
        // charge density
        idn->rhon[i][j][k] = 0.0;  // quantities defined on node
        // current
        idn->Jx[i][j][k] = 0.0;  // quantities defined on node
        idn->Jy[i][j][k] = 0.0;  // quantities defined on node
        idn->Jz[i][j][k] = 0.0;  // quantities defined on node
        // pressure
        idn->pxx[i][j][k] = 0.0;  // quantities defined on node
        idn->pxy[i][j][k] = 0.0;  // quantities defined on node
        idn->pxz[i][j][k] = 0.0;  // quantities defined on node
        idn->pyy[i][j][k] = 0.0;  // quantities defined on node
        idn->pyz[i][j][k] = 0.0;  // quantities defined on node
        idn->pzz[i][j][k] = 0.0;  // quantities defined on node
      }

      // center cell rhoc
#pragma omp parallel for
  for (int i = 0; i < grd->nxc; i++)
    for (int j = 0; j < grd->nyc; j++)
      for (int k = 0; k < grd->nzc; k++) {
        idn->rhoc[i][j][k] = 0.0;  // quantities defined on center cells
      }
}

/** set all the densities to zero */
void setZeroDensities(struct interpDensNet *idn, struct interpDensSpecies *ids,
                      struct grid *grd, int ns) {
  setZeroNetDensities(idn, grd);
  setZeroSpeciesDensities(ids, grd, ns);
}

/** set all the densities to zero */
void sumOverSpecies(struct interpDensNet *idn, struct interpDensSpecies *ids,
                    struct grid *grd, int ns) {
  for (int is = 0; is < ns; is++)
    for (int i = 0; i < grd->nxn; i++)
      for (int j = 0; j < grd->nyn; j++)
        for (int k = 0; k < grd->nzn; k++) {
          // density
          idn->rhon[i][j][k] += ids[is].rhon[i][j][k];

          // These are not really needed for the algoritm
          // They might needed for the algorithm
          // J
          idn->Jx[i][j][k] += ids[is].Jx[i][j][k];
          idn->Jy[i][j][k] += ids[is].Jy[i][j][k];
          idn->Jz[i][j][k] += ids[is].Jz[i][j][k];
          // pressure
          idn->pxx[i][j][k] += ids[is].pxx[i][j][k];
          idn->pxy[i][j][k] += ids[is].pxy[i][j][k];
          idn->pxz[i][j][k] += ids[is].pxz[i][j][k];
          idn->pyy[i][j][k] += ids[is].pyy[i][j][k];
          idn->pyz[i][j][k] += ids[is].pyz[i][j][k];
          idn->pzz[i][j][k] += ids[is].pzz[i][j][k];
        }
}
