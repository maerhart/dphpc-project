#include "MaxwellSolver.h"

void MaxwellSource(FPfield *bkrylov, struct grid *grd, struct EMfield *field,
                   struct EMfield_aux *field_aux, struct interpDens_aux *id_aux,
                   struct parameters *param) {
  // get drid point nodes
  int nxn = grd->nxn;
  int nyn = grd->nyn;
  int nzn = grd->nzn;

  FPfield delt = param->c * param->th * param->dt;

  // temporary arrays
  FPfield ***tempX = (FPfield***) newArr(sizeof(FPfield), 3, nxn, nyn, nzn);
  FPfield ***tempY = (FPfield***) newArr(sizeof(FPfield), 3, nxn, nyn, nzn);
  FPfield ***tempZ = (FPfield***) newArr(sizeof(FPfield), 3, nxn, nyn, nzn);
  FPfield ***tempXN = (FPfield***) newArr(sizeof(FPfield), 3, nxn, nyn, nzn);
  FPfield ***tempYN = (FPfield***) newArr(sizeof(FPfield), 3, nxn, nyn, nzn);
  FPfield ***tempZN = (FPfield***) newArr(sizeof(FPfield), 3, nxn, nyn, nzn);
  FPfield ***temp2X = (FPfield***) newArr(sizeof(FPfield), 3, nxn, nyn, nzn);
  FPfield ***temp2Y = (FPfield***) newArr(sizeof(FPfield), 3, nxn, nyn, nzn);
  FPfield ***temp2Z = (FPfield***) newArr(sizeof(FPfield), 3, nxn, nyn, nzn);

  // set to zero
  eqValue3(0.0, tempX, nxn, nyn, nzn);
  eqValue3(0.0, tempY, nxn, nyn, nzn);
  eqValue3(0.0, tempZ, nxn, nyn, nzn);
  eqValue3(0.0, tempXN, nxn, nyn, nzn);
  eqValue3(0.0, tempYN, nxn, nyn, nzn);
  eqValue3(0.0, tempZN, nxn, nyn, nzn);
  eqValue3(0.0, temp2X, nxn, nyn, nzn);
  eqValue3(0.0, temp2Y, nxn, nyn, nzn);
  eqValue3(0.0, temp2Z, nxn, nyn, nzn);
  // communicate

  // fixBforcefree(grid,vct);
  // fixBgem(grid, vct);

  // prepare curl of B for known term of Maxwell solver: for the source term
  applyBCscalarFieldC(field_aux->Bxc, grd, param);  // tangential Neumann
  applyBCscalarFieldC(field_aux->Byc, grd, param);  // normal is zero
  applyBCscalarFieldC(field_aux->Bzc, grd, param);  // tangential Neumann

  curlC2N(tempXN, tempYN, tempZN, field_aux->Bxc, field_aux->Byc,
          field_aux->Bzc, grd);

  scale3_into(temp2X, id_aux->Jxh, -param->fourpi / param->c, nxn, nyn, nzn);
  scale3_into(temp2Y, id_aux->Jyh, -param->fourpi / param->c, nxn, nyn, nzn);
  scale3_into(temp2Z, id_aux->Jzh, -param->fourpi / param->c, nxn, nyn, nzn);

  sum3(temp2X, tempXN, nxn, nyn, nzn);
  sum3(temp2Y, tempYN, nxn, nyn, nzn);
  sum3(temp2Z, tempZN, nxn, nyn, nzn);
  scale3(temp2X, delt, nxn, nyn, nzn);
  scale3(temp2Y, delt, nxn, nyn, nzn);
  scale3(temp2Z, delt, nxn, nyn, nzn);

  // communicateCenterBC_P(nxc, nyc, nzc, rhoh, 2, 2, 2, 2, 2, 2, vct);
  applyBCscalarDensC(id_aux->rhoh, grd, param);
  gradC2N(tempX, tempY, tempZ, id_aux->rhoh, grd);

  scale3(tempX, -delt * delt * param->fourpi, nxn, nyn, nzn);
  scale3(tempY, -delt * delt * param->fourpi, nxn, nyn, nzn);
  scale3(tempZ, -delt * delt * param->fourpi, nxn, nyn, nzn);

  // sum E, past values
  sum3(tempX, field->Ex, nxn, nyn, nzn);
  sum3(tempY, field->Ey, nxn, nyn, nzn);
  sum3(tempZ, field->Ez, nxn, nyn, nzn);

  // sum curl(B) + jhat part
  sum3(tempX, temp2X, nxn, nyn, nzn);
  sum3(tempY, temp2Y, nxn, nyn, nzn);
  sum3(tempZ, temp2Z, nxn, nyn, nzn);

  // physical space -> Krylov space
  phys2solver3(bkrylov, tempX, tempY, tempZ, nxn, nyn, nzn);

  delArr(3, tempX);
  delArr(3, tempY);
  delArr(3, tempZ);

  delArr(3, tempXN);
  delArr(3, tempYN);
  delArr(3, tempZN);

  delArr(3, temp2X);
  delArr(3, temp2Y);
  delArr(3, temp2Z);
}

/** Maxwell Image */
void MaxwellImage(FPfield *im, FPfield *vector, struct EMfield *field,
                  struct interpDensSpecies *ids, struct grid *grd, struct parameters *param) {
  FPfield beta, edotb, omcx, omcy, omcz, denom;
  FPfield delt = param->c * param->th * param->dt;

  // get drid point nodes
  int nxn = grd->nxn;
  int nyn = grd->nyn;
  int nzn = grd->nzn;

  // get number of cells
  int nxc = grd->nxc;
  int nyc = grd->nyc;
  int nzc = grd->nzc;

  // allocate temporary arrays
  FPfield ***vectX = (FPfield ***) newArr(sizeof(FPfield), 3, nxn, nyn, nzn);
  FPfield ***vectY = (FPfield ***) newArr(sizeof(FPfield), 3, nxn, nyn, nzn);
  FPfield ***vectZ = (FPfield ***) newArr(sizeof(FPfield), 3, nxn, nyn, nzn);
  FPfield ***imageX = (FPfield ***) newArr(sizeof(FPfield), 3, nxn, nyn, nzn);
  FPfield ***imageY = (FPfield ***) newArr(sizeof(FPfield), 3, nxn, nyn, nzn);
  FPfield ***imageZ = (FPfield ***) newArr(sizeof(FPfield), 3, nxn, nyn, nzn);
  FPfield ***Dx = (FPfield ***) newArr(sizeof(FPfield), 3, nxn, nyn, nzn);
  FPfield ***Dy = (FPfield ***) newArr(sizeof(FPfield), 3, nxn, nyn, nzn);
  FPfield ***Dz = (FPfield ***) newArr(sizeof(FPfield), 3, nxn, nyn, nzn);
  FPfield ***tempX = (FPfield ***) newArr(sizeof(FPfield), 3, nxn, nyn, nzn);
  FPfield ***tempY = (FPfield ***) newArr(sizeof(FPfield), 3, nxn, nyn, nzn);
  FPfield ***tempZ = (FPfield ***) newArr(sizeof(FPfield), 3, nxn, nyn, nzn);

  FPfield ***divC = (FPfield ***) newArr(sizeof(FPfield), 3, nxc, nyc, nzc);

// set to zero
#pragma omp parallel for
  for (int i = 0; i < nxn; i++)
    for (int j = 0; j < nyn; j++)
#pragma clang loop vectorize(enable)
      for (int k = 0; k < nzn; k++) {
        vectX[i][j][k] = 0.0;
        vectY[i][j][k] = 0.0;
        vectZ[i][j][k] = 0.0;
        imageX[i][j][k] = 0.0;
        imageY[i][j][k] = 0.0;
        imageZ[i][j][k] = 0.0;
        Dx[i][j][k] = 0.0;
        Dy[i][j][k] = 0.0;
        Dz[i][j][k] = 0.0;
        tempX[i][j][k] = 0.0;
        tempY[i][j][k] = 0.0;
        tempZ[i][j][k] = 0.0;
      }

  // move from krylov space to physical space
  solver2phys3(vectX, vectY, vectZ, vector, nxn, nyn, nzn);

  // here we need to impose BC on E: before Laplacian

  ////////
  ////
  //// This part needs to be fixeds. Put PEC
  ///  This puts zero also on the electric field normal
  ///
  //////

  applyBCscalarFieldNzero(vectX, grd, param);
  applyBCscalarFieldNzero(vectY, grd, param);
  applyBCscalarFieldNzero(vectZ, grd, param);

  ////////
  ////
  ////
  ///
  //////

  lapN2N_V(imageX, vectX, grd);
  lapN2N_V(imageY, vectY, grd);
  lapN2N_V(imageZ, vectZ, grd);
  neg3(imageX, nxn, nyn, nzn);
  neg3(imageY, nxn, nyn, nzn);
  neg3(imageZ, nxn, nyn, nzn);

  // grad(div(mu dot E(n + theta)) mu dot E(n + theta) = D
  for (int is = 0; is < param->ns; is++) {
    beta = .5 * param->qom[is] * param->dt / param->c;
#pragma omp parallel for private(omcx, omcy, omcz, edotb, denom)
    for (int i = 0; i < nxn; i++)
      for (int j = 0; j < nyn; j++)
#pragma clang loop vectorize(enable)
        for (int k = 0; k < nzn; k++) {
          omcx = beta * field->Bxn[i][j][k];
          omcy = beta * field->Byn[i][j][k];
          omcz = beta * field->Bzn[i][j][k];
          edotb = vectX[i][j][k] * omcx + vectY[i][j][k] * omcy +
                  vectZ[i][j][k] * omcz;
          denom = param->fourpi / 2 * delt * param->dt / param->c *
                  param->qom[is] * ids[is].rhon[i][j][k] /
                  (1.0 + omcx * omcx + omcy * omcy + omcz * omcz);
          Dx[i][j][k] +=
              (vectX[i][j][k] +
               (vectY[i][j][k] * omcz - vectZ[i][j][k] * omcy + edotb * omcx)) *
              denom;
          Dy[i][j][k] +=
              (vectY[i][j][k] +
               (vectZ[i][j][k] * omcx - vectX[i][j][k] * omcz + edotb * omcy)) *
              denom;
          Dz[i][j][k] +=
              (vectZ[i][j][k] +
               (vectX[i][j][k] * omcy - vectY[i][j][k] * omcx + edotb * omcz)) *
              denom;
        }
  }

  // apply boundary condition to Dx, Dy and Dz
  applyBCscalarFieldNzero(Dx, grd, param);
  applyBCscalarFieldNzero(Dy, grd, param);
  applyBCscalarFieldNzero(Dz, grd, param);

  divN2C(divC, Dx, Dy, Dz, grd);

  // communicate you should put BC

  applyBCscalarFieldC(divC, grd, param);

  gradC2N(tempX, tempY, tempZ, divC, grd);

  // -lap(E(n +theta)) - grad(div(mu dot E(n + theta))
  sub3(imageX, tempX, nxn, nyn, nzn);
  sub3(imageY, tempY, nxn, nyn, nzn);
  sub3(imageZ, tempZ, nxn, nyn, nzn);

  // scale delt*delt
  scale3(imageX, delt * delt, nxn, nyn, nzn);
  scale3(imageY, delt * delt, nxn, nyn, nzn);
  scale3(imageZ, delt * delt, nxn, nyn, nzn);

  // -lap(E(n +theta)) - grad(div(mu dot E(n + theta)) + eps dot E(n + theta)
  sum3(imageX, Dx, nxn, nyn, nzn);
  sum3(imageY, Dy, nxn, nyn, nzn);
  sum3(imageZ, Dz, nxn, nyn, nzn);
  sum3(imageX, vectX, nxn, nyn, nzn);
  sum3(imageY, vectY, nxn, nyn, nzn);
  sum3(imageZ, vectZ, nxn, nyn, nzn);

  // move from physical space to krylov space
  phys2solver3(im, imageX, imageY, imageZ, nxn, nyn, nzn);

  // deallocate
  delArr(3, vectX);
  delArr(3, vectY);
  delArr(3, vectZ);
  delArr(3, imageX);
  delArr(3, imageY);
  delArr(3, imageZ);
  delArr(3, Dx);
  delArr(3, Dy);
  delArr(3, Dz);
  delArr(3, tempX);
  delArr(3, tempY);
  delArr(3, tempZ);

  delArr(3, divC);
}

/** calculate the electric field using second order curl-curl formulation of
 * Maxwell equations */
void calculateE(struct grid *grd, struct EMfield_aux *field_aux, struct EMfield *field,
                struct interpDens_aux *id_aux, struct interpDensSpecies *ids,
                struct parameters *param) {
  // get frid points
  int nxn = grd->nxn;
  int nyn = grd->nyn;
  int nzn = grd->nzn;

  // int nxc = grd->nxc;
  // int nyc = grd->nyc;
  // int nzc = grd->nzc;

  // krylov vectors
  FPfield *xkrylov = (FPfield*) newArr(sizeof(FPfield), 1, 3 * (nxn - 2) * (nyn - 2) * (nzn - 2));
      // new FPfield[3 * (nxn - 2) * (nyn - 2) * (nzn - 2)];  // 3 E components
  FPfield *bkrylov = (FPfield*) newArr(sizeof(FPfield), 1, 3 * (nxn - 2) * (nyn - 2) * (nzn - 2));
      // new FPfield[3 * (nxn - 2) * (nyn - 2) * (nzn - 2)];  // 3 components: source

  eqValue1(0.0, xkrylov, 3 * (nxn - 2) * (nyn - 2) * (nzn - 2));
  eqValue1(0.0, bkrylov, 3 * (nxn - 2) * (nyn - 2) * (nzn - 2));

  printf("*** MAXWELL SOLVER ***\n");

  // form the source term
  MaxwellSource(bkrylov, grd, field, field_aux, id_aux, param);

  // prepare xkrylov solver
  phys2solver3(xkrylov, field->Ex, field->Ey, field->Ez, nxn, nyn, nzn);

  // call the GMRes
  GMRes(&MaxwellImage, xkrylov, 3 * (nxn - 2) * (nyn - 2) * (nzn - 2), bkrylov,
        20, 200, param->GMREStol, field, ids, grd, param);

  // move from krylov space to physical space
  solver2phys3(field_aux->Exth, field_aux->Eyth, field_aux->Ezth, xkrylov, nxn,
              nyn, nzn);

  // add E from Eth
  addscale3(1 / param->th, -(1.0 - param->th) / param->th, field->Ex,
           field_aux->Exth, nxn, nyn, nzn);
  addscale3(1 / param->th, -(1.0 - param->th) / param->th, field->Ey,
           field_aux->Eyth, nxn, nyn, nzn);
  addscale3(1 / param->th, -(1.0 - param->th) / param->th, field->Ez,
           field_aux->Ezth, nxn, nyn, nzn);

  // Smooth: You might have special smoothing for E (imposing E on the BC)
  // smoothInterpScalarN(field_aux->Exth, grd, param);
  // smoothInterpScalarN(field_aux->Eyth, grd, param);
  // smoothInterpScalarN(field_aux->Ezth, grd, param);

  // smoothInterpScalarN(field->Ex, grd, param);
  // smoothInterpScalarN(field->Ey, grd, param);
  // smoothInterpScalarN(field->Ez, grd, param);

  // deallocate temporary arrays
  delArr(1, xkrylov);
  delArr(1, bkrylov);
}

// calculate the magnetic field from Faraday's law
void calculateB(struct grid *grd, struct EMfield_aux *field_aux, struct EMfield *field,
                struct parameters *param) {
  int nxc = grd->nxc;
  int nyc = grd->nyc;
  int nzc = grd->nzc;

  FPfield ***tempXC = (FPfield***) newArr(sizeof(FPfield), 3, nxc, nyc, nzc);
  FPfield ***tempYC = (FPfield***) newArr(sizeof(FPfield), 3, nxc, nyc, nzc);
  FPfield ***tempZC = (FPfield***) newArr(sizeof(FPfield), 3, nxc, nyc, nzc);

  printf("*** B CALCULATION ***\n");

  // calculate the curl of Eth
  curlN2C(tempXC, tempYC, tempZC, field_aux->Exth, field_aux->Eyth,
          field_aux->Ezth, grd);
  // update the magnetic field
  addscale3(-param->c * param->dt, 1, field_aux->Bxc, tempXC, nxc, nyc, nzc);
  addscale3(-param->c * param->dt, 1, field_aux->Byc, tempYC, nxc, nyc, nzc);
  addscale3(-param->c * param->dt, 1, field_aux->Bzc, tempZC, nxc, nyc, nzc);

  applyBCscalarFieldC(field_aux->Bxc, grd, param);
  applyBCscalarFieldC(field_aux->Byc, grd, param);
  applyBCscalarFieldC(field_aux->Bzc, grd, param);

  // interpolate C2N

  interpC2Nfield(field->Bxn, field_aux->Bxc, grd);
  interpC2Nfield(field->Byn, field_aux->Byc, grd);
  interpC2Nfield(field->Bzn, field_aux->Bzc, grd);

  // BC on By ???

  // deallocate
  delArr(3, tempXC);
  delArr(3, tempYC);
  delArr(3, tempZC);
}

/* Poisson Image */
void PoissonImage(FPfield *image, FPfield *vector, struct grid *grd) {
  // center cells
  int nxc = grd->nxc;
  int nyc = grd->nyc;
  int nzc = grd->nzc;

  // allocate temporary arrays
  FPfield ***temp = (FPfield***) newArr(sizeof(FPfield), 3, nxc, nyc, nzc);
  FPfield ***im = (FPfield***) newArr(sizeof(FPfield), 3, nxc, nyc, nzc);

  // set arrays to zero
  for (int i = 0; i < (nxc - 2) * (nyc - 2) * (nzc - 2); i++) image[i] = 0.0;
  for (int i = 0; i < nxc; i++)
    for (int j = 0; j < nyc; j++)
      for (int k = 0; k < nzc; k++) {
        temp[i][j][k] = 0.0;
        im[i][j][k] = 0.0;
      }
  solver2phys1(temp, vector, nxc, nyc, nzc);
  // the BC for this Laplacian are zero on th eboundary?
  lapC2C(im, temp, grd);
  // move from physical space to krylov space
  phys2solver1(image, im, nxc, nyc, nzc);

// for(int iii=200; iii<220; iii++){
//     printf("%d : %f\n", iii, im[iii]);
//   }

  // deallocate temporary array and objects
  delArr(3, temp);
  delArr(3, im);
}

/** calculate Poisson Correction */
void divergenceCleaning(struct grid *grd, struct EMfield_aux *field_aux, struct EMfield *field,
                        struct interpDensNet *idn, struct parameters *param) {
  // get the number of cells
  int nxc = grd->nxc;
  int nyc = grd->nyc;
  int nzc = grd->nzc;
  //  get the number of nodes
  int nxn = grd->nxn;
  int nyn = grd->nyn;
  int nzn = grd->nzn;

  // temporary array for div(E)
  FPfield ***divE = (FPfield***) newArr(sizeof(FPfield), 3, nxc, nyc, nzc);
  FPfield ***gradPHIX = (FPfield***) newArr(sizeof(FPfield), 3, nxn, nyn, nzn);
  FPfield ***gradPHIY = (FPfield***) newArr(sizeof(FPfield), 3, nxn, nyn, nzn);
  FPfield ***gradPHIZ = (FPfield***) newArr(sizeof(FPfield), 3, nxn, nyn, nzn);

  // 1D vectors for solver
  FPfield *xkrylovPoisson = (FPfield*) newArr(sizeof(FPfield), 1, (nxc - 2) * (nyc - 2) * (nzc - 2));
  FPinterp *bkrylovPoisson = (FPinterp*) newArr(sizeof(FPinterp), 1, (nxc - 2) * (nyc - 2) * (nzc - 2));
  // set to zero xkrylov and bkrylov
  for (int i = 0; i < ((nxc - 2) * (nyc - 2) * (nzc - 2)); i++) {
    xkrylovPoisson[i] = 0.0;
    bkrylovPoisson[i] = 0.0;
  }
  for (int i = 0; i < nxn; i++)
    for (int j = 0; j < nyn; j++)
      for (int k = 0; k < nzn; k++) {
        gradPHIX[i][j][k] = 0.0;
        gradPHIY[i][j][k] = 0.0;
        gradPHIZ[i][j][k] = 0.0;
      }

  printf("*** DIVERGENCE CLEANING ***\n");
  divN2C(divE, field->Ex, field->Ey, field->Ez, grd);

  // int offset = 30;
  // int len=5;
  // for(int i=offset; i<offset+len; i++){
  //   for(int j=offset; j<offset+len; j++){
  //     for(int k=offset; k<offset+len; k++){
  //       printf("%d,%d,%d : %.8f\n", i, j, k, divE[i][j][k]);
  //     }
  //   }
  // }

  for (int i = 0; i < nxc; i++)
    for (int j = 0; j < nyc; j++)
      for (int k = 0; k < nzc; k++){
        divE[i][j][k] = divE[i][j][k] - param->fourpi * idn->rhoc[i][j][k];
        field_aux->Phi[i][j][k] = 0.0;
      }

  // phys to solver
  phys2solver1(bkrylovPoisson, divE, nxc, nyc, nzc);


  // call CG for solving Poisson
  if (!CG(xkrylovPoisson, (nxc - 2) * (nyc - 2) * (nzc - 2), bkrylovPoisson,
          3000, param->CGtol, &PoissonImage, grd))
    printf("*ERROR - CG not Converged\n");

  solver2phys1(field_aux->Phi, xkrylovPoisson, nxc, nyc, nzc);

  // This has Newmann. If commented has zero in ghost cells
  applyBCscalarFieldC(field_aux->Phi, grd, param);
  gradC2N(gradPHIX, gradPHIY, gradPHIZ, field_aux->Phi, grd);

  for (int i = 0; i < nxn; i++)
    for (int j = 0; j < nyn; j++)
      for (int k = 0; k < nzn; k++) {
        field->Ex[i][j][k] -= gradPHIX[i][j][k];
        field->Ey[i][j][k] -= gradPHIY[i][j][k];
        field->Ez[i][j][k] -= gradPHIZ[i][j][k];
      }

  // deallocate vectors
  delArr(1,xkrylovPoisson);
  delArr(1,bkrylovPoisson);

  // deallocate temporary array
  delArr(3, divE);
  delArr(3, gradPHIX);
  delArr(3, gradPHIY);
  delArr(3, gradPHIZ);
}

/** Calculate hat densities: Jh and rhoh*/
void calculateHatDensities(struct interpDens_aux *id_aux,
                           struct interpDensNet *idn,
                           struct interpDensSpecies *ids, struct EMfield *field,
                           struct grid *grd, struct parameters *param) {
  // parameters
  FPfield beta, edotb, omcx, omcy, omcz, denom;

  // nodes
  int nxn = grd->nxn;
  int nyn = grd->nyn;
  int nzn = grd->nzn;

  // centers
  int nxc = grd->nxc;
  int nyc = grd->nyc;
  int nzc = grd->nzc;

  // allocate temporary ararys
  // center
  FPinterp ***tempXC = (FPinterp***) newArr(sizeof(FPinterp), 3, nxc, nyc, nzc);
  FPinterp ***tempYC = (FPinterp***) newArr(sizeof(FPinterp), 3, nxc, nyc, nzc);
  FPinterp ***tempZC = (FPinterp***) newArr(sizeof(FPinterp), 3, nxc, nyc, nzc);
  // nodes
  FPinterp ***tempXN = (FPinterp***) newArr(sizeof(FPinterp), 3, nxn, nyn, nzn);
  FPinterp ***tempYN = (FPinterp***) newArr(sizeof(FPinterp), 3, nxn, nyn, nzn);
  FPinterp ***tempZN = (FPinterp***) newArr(sizeof(FPinterp), 3, nxn, nyn, nzn);

  // Set J hat to zero: Important to set to zero because then it accumulate in
  // PIdot

  for (int i = 0; i < nxn; i++)
    for (int j = 0; j < nyn; j++)
      for (int k = 0; k < nzn; k++) {
        id_aux->Jxh[i][j][k] = 0.0;
        id_aux->Jyh[i][j][k] = 0.0;
        id_aux->Jzh[i][j][k] = 0.0;
      }
  for (int i = 0; i < nxc; i++)
    for (int j = 0; j < nyc; j++)
      for (int k = 0; k < nzc; k++) 
        id_aux->rhoh[i][j][k] = 0.0;

  // smoothing of rhoc: BC have applied before returning
  // smoothInterpScalarC(idn->rhoc,grd,param);
  // apply boundary to rhoh
  applyBCscalarDensC(id_aux->rhoh, grd,
                     param);  // set BC on ghost cells before interp

  for (int is = 0; is < param->ns; is++) {
    divSymmTensorN2C(tempXC, tempYC, tempZC, ids[is].pxx, ids[is].pxy,
                     ids[is].pxz, ids[is].pyy, ids[is].pyz, ids[is].pzz, grd);

    // scale the pressure tensor
    scale3(tempXC, -param->dt / 2.0, nxc, nyc, nzc);
    scale3(tempYC, -param->dt / 2.0, nxc, nyc, nzc);
    scale3(tempZC, -param->dt / 2.0, nxc, nyc, nzc);

    // apply BC to centers before interpolation: this is not needed
    applyBCscalarDensC(tempXC, grd,
                       param);  // set BC on ghost cells before interp
    applyBCscalarDensC(tempYC, grd,
                       param);  // set BC on ghost cells before interp
    applyBCscalarDensC(tempZC, grd,
                       param);  // set BC on ghost cells before interp

    // interpolation
    interpC2Ninterp(tempXN, tempXC, grd);
    interpC2Ninterp(tempYN, tempYC, grd);
    interpC2Ninterp(tempZN, tempZC, grd);

    // sum(tempXN, Jxs, nxn, nyn, nzn, is); sum(tempYN, Jys, nxn, nyn, nzn, is);
    // sum(tempZN, Jzs, nxn, nyn, nzn, is);
    for (int i = 0; i < nxn; i++)
      for (int j = 0; j < nyn; j++)
        for (int k = 0; k < nzn; k++) {
          tempXN[i][j][k] += ids[is].Jx[i][j][k];
          tempYN[i][j][k] += ids[is].Jy[i][j][k];
          tempZN[i][j][k] += ids[is].Jz[i][j][k];
        }

    // PIDOT //PIdot(Jxh, Jyh, Jzh, tempXN, tempYN, tempZN, is, grid);
    beta = .5 * param->qom[is] * param->dt / param->c;
    for (int i = 0; i < nxn; i++)
      for (int j = 0; j < nyn; j++)
        for (int k = 0; k < nzn; k++) {
          omcx = beta * field->Bxn[i][j][k];
          omcy = beta * field->Byn[i][j][k];
          omcz = beta * field->Bzn[i][j][k];
          edotb = tempXN[i][j][k] * omcx + tempYN[i][j][k] * omcy +
                  tempZN[i][j][k] * omcz;
          denom = 1 / (1.0 + omcx * omcx + omcy * omcy + omcz * omcz);
          id_aux->Jxh[i][j][k] +=
              (tempXN[i][j][k] + (tempYN[i][j][k] * omcz -
                                  tempZN[i][j][k] * omcy + edotb * omcx)) *
              denom;
          id_aux->Jyh[i][j][k] +=
              (tempYN[i][j][k] + (tempZN[i][j][k] * omcx -
                                  tempXN[i][j][k] * omcz + edotb * omcy)) *
              denom;
          id_aux->Jzh[i][j][k] +=
              (tempZN[i][j][k] + (tempXN[i][j][k] * omcy -
                                  tempYN[i][j][k] * omcx + edotb * omcz)) *
              denom;
        }
  }

  // smooth J hat
  // smoothInterpScalarN(id_aux->Jxh,grd,param);
  // smoothInterpScalarN(id_aux->Jyh,grd,param);
  // smoothInterpScalarN(id_aux->Jzh,grd,param);

  // calculate rho hat = rho - (dt*theta)div(jhat)
  // set tempXC to zero
  for (int i = 0; i < nxc; i++)
    for (int j = 0; j < nyc; j++)
      for (int k = 0; k < nzc; k++) tempXC[i][j][k] = 0.0;

  // in principle dont need this
  applyBCscalarDensN(id_aux->Jxh, grd, param);
  applyBCscalarDensN(id_aux->Jyh, grd, param);
  applyBCscalarDensN(id_aux->Jzh, grd, param);

  divN2C(tempXC, id_aux->Jxh, id_aux->Jyh, id_aux->Jzh, grd);

  for (int i = 0; i < nxc; i++)
    for (int j = 0; j < nyc; j++)
      for (int k = 0; k < nzc; k++)
        id_aux->rhoh[i][j][k] =
            idn->rhoc[i][j][k] - param->dt * param->th * tempXC[i][j][k];

  // apply boundary to rhoh
  applyBCscalarDensC(id_aux->rhoh, grd,
                     param);  // set BC on ghost cells before interp

  // deallocate
  delArr(3, tempXC);
  delArr(3, tempYC);
  delArr(3, tempZC);
  delArr(3, tempXN);
  delArr(3, tempYN);
  delArr(3, tempZN);
}
