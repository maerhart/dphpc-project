#include "Solvers.h"
#include <math.h>
#include "Basic.h"
#include <string.h>

bool CG(FPfield *xkrylov, int xkrylovlen, FPfield *b, int maxit, double tol,
        GENERIC_IMAGE FunctionImage, struct grid *grd) {
  // allocate residual, image, p, b, calculated on central points

  FPfield *r = (FPfield*) newArr(sizeof(FPfield), 1, xkrylovlen);
  FPfield *v = (FPfield*) newArr(sizeof(FPfield), 1, xkrylovlen);
  FPfield *z = (FPfield*) newArr(sizeof(FPfield), 1, xkrylovlen);
  FPfield *im = (FPfield*) newArr(sizeof(FPfield), 1, xkrylovlen);


  // aux variables
  FPfield c = 0.0, t = 0.0, d = 0.0, initial_error = 0.0, dotP = 0.0;

  bool CONVERGED = false;
  bool CGVERBOSE = false;

  // i is the CG counter
  // don't use it for loops
  int i = 0;

  // initial guess for x: all the components are equal to 0
  for (int ii = 0; ii < xkrylovlen; ii++) xkrylov[ii] = 0.0;

  // Compute r = b -Ax
  // sub(r, b, im, xkrylovlen);
  // v = r
  // eq(v, r, xkrylovlen);
  // c = dotP(r, r, xkrylovlen);

  (*FunctionImage)(im, xkrylov, grd);

  c = 0.0;
  for (int ii = 0; ii < xkrylovlen; ii++) {
    r[ii] = b[ii] - im[ii];
    v[ii] = r[ii];
    c += r[ii] * r[ii];
  }
  initial_error = sqrt(c);
  printf("Initial error: %f\n", initial_error);

  if (CGVERBOSE) {

    printf("------------------------------------\n");
    printf("-               CG                 -\n");
    printf("------------------------------------\n\n");

  }

  while (i < maxit) {
    // claculate image of Poisson equation
    (*FunctionImage)(z, v, grd);

    // t = c / dotP(v, z, xkrylovlen);
    dotP = 0.0;
    for (int ii = 0; ii < xkrylovlen; ii++) dotP += v[ii] * z[ii];
    t = c / dotP;

    // x(i+1) = x + t*v - addscale(t, xkrylov, v, xkrylovlen);
    for (int ii = 0; ii < xkrylovlen; ii++) xkrylov[ii] += t * v[ii];

    // r(i+1) = r - t*z - addscale(-t, r, z, xkrylovlen);
    for (int ii = 0; ii < xkrylovlen; ii++) r[ii] -= t * z[ii];

    // d = dotP(r, r, xkrylovlen);
    d = 0.0;
    for (int ii = 0; ii < xkrylovlen; ii++) d += r[ii] * r[ii];

    if (CGVERBOSE)
      printf("Iteration # %d - norm of residual relative to initial error: %f\n", 
        i, sqrt(d) / initial_error);

    if (sqrt(d) < tol * initial_error) {
      printf("CG converged at iteration # %d\n", i);
      CONVERGED = true;
      break;
    } else if (sqrt(d) > 10E8 * initial_error) {
      // HERE
      fprintf(stderr, "CG not converging");
      fprintf(stderr, "CG stopped");

      CONVERGED = false;
      break;
    }

    // addscale(1, d/c, v, r, xkrylovlen); DOUBLE CHECK THIS
    for (int ii = 0; ii < xkrylovlen; ii++) v[ii] = v[ii] * d / c + r[ii];

    c = d;
    // update the CG iteration counter
    i++;
  }

  // deallocate

  delArr(1, r);
  delArr(1, v);
  delArr(1, z);
  delArr(1, im);

  // return true if converged
  return (CONVERGED);
}

void GMRes(GENERIC_IMAGE_GMRES FunctionImage, FPfield *xkrylov, int xkrylovlen,
           FPfield *b, int m, int max_iter, double tol, struct EMfield *field,
          struct interpDensSpecies *ids, struct grid *grd, struct parameters *param) {
  // check size of m
  if (m > xkrylovlen) {
    fprintf(stderr, "In GMRES the dimension of Krylov space(m) can't be > (length of krylov vector)/(# processors)\n");
    return;
  }

  bool GMRESVERBOSE = false;

  FPfield initial_error, normb, rho_tol, av, mu, htmp, tmp, delta = 0.001;

  // arrays for GMRes
  FPfield *r = (FPfield*) newArr(sizeof(FPfield), 1, xkrylovlen);
  FPfield *v = (FPfield*) newArr(sizeof(FPfield), 1, xkrylovlen);
  FPfield *w = (FPfield*) newArr(sizeof(FPfield), 1, xkrylovlen);
  FPfield *im = (FPfield*) newArr(sizeof(FPfield), 1, xkrylovlen);

  FPfield *s = (FPfield*) newArr(sizeof(FPfield), 1, m+1);
  FPfield *cs = (FPfield*) newArr(sizeof(FPfield), 1, m+1);
  FPfield *sn = (FPfield*) newArr(sizeof(FPfield), 1, m+1);
  FPfield *y = (FPfield*) newArr(sizeof(FPfield), 1, m+1);

  // eqValue(0.0, s, m + 1);
  // eqValue(0.0, cs, m + 1);
  // eqValue(0.0, sn, m + 1);
  // eqValue(0.0, y, m + 1);

  // allocate H for storing the results from decomposition
  FPfield **H = (FPfield**) newArr(sizeof(FPfield), 2, m + 1, m);
  for (int ii = 0; ii < m + 1; ii++)
    for (int jj = 0; jj < m; jj++) H[ii][jj] = 0;

  // allocate V
  FPfield **V = (FPfield**) newArr(sizeof(FPfield), 2, xkrylovlen, m + 1);
  for (int ii = 0; ii < xkrylovlen; ii++)
    for (int jj = 0; jj < m + 1; jj++) V[ii][jj] = 0;

  if (GMRESVERBOSE) {
    printf("------------------------------------\n");
    printf("-             GMRES                -\n");
    printf("------------------------------------\n\n");
  }

  for (int itr = 0; itr < max_iter; itr++) {
    // r = b - A*x
    (*FunctionImage)(im, xkrylov, field, ids, grd, param);
    sub1(r, b, im, xkrylovlen);

    // error as norm of th 1residual
    initial_error = normP(r, xkrylovlen);
    normb = normP(b, xkrylovlen);
    if (normb < 1E-16) {
      // normb = 1.0;
      // initial_error = 0.0;
    }
    if (itr == 0) {
      printf("Initial residual: %f, norm b vector (source) = %f\n", initial_error, normb);
      rho_tol = initial_error * tol;

      if ((initial_error / normb) <= tol) {
        printf("GMRES converged without iterations: initial error < tolerance\n");

        delArr(1, r);
        delArr(1, im);
        delArr(1, s);
        delArr(1, v);
        delArr(1, cs);
        delArr(1, sn);
        delArr(1, w);
        delArr(1, y);
        delArr(2, H);
        delArr(2, V);

        return;
      }
    }

    scale1_into(v, r, (1.0 / initial_error), xkrylovlen);
    putColumn(V, v, 0, xkrylovlen);
    eqValue1(0.0, s, m + 1);
    s[0] = initial_error;
    int k = 0;

    // GMRes iteration
    while (rho_tol < initial_error && k < m) {
      // w= A*V(:,k)  );
      getColumn(v, V, k, xkrylovlen);
      (*FunctionImage)(w, v, field, ids, grd, param);
      putColumn(V, w, k + 1, xkrylovlen);
      av = normP(w, xkrylovlen);

      for (int j = 0; j <= k; j++) {
        getColumn(v, V, j, xkrylovlen);
        H[j][k] = dot(w, v, xkrylovlen);
        addscale(-H[j][k], 1.0, w, v, xkrylovlen);
      }
      putColumn(V, w, k + 1, xkrylovlen);
      H[k + 1][k] = normP(w, xkrylovlen);

      if (av + delta * H[k + 1][k] == av) {
        for (int j = 0; j <= k; j++) {
          getColumn(v, V, j, xkrylovlen);
          htmp = dot(w, v, xkrylovlen);
          H[j][k] = H[j][k] + htmp;
          addscale(-htmp, 1.0, w, v, xkrylovlen);
        }
        putColumn(V, w, k + 1, xkrylovlen);

        H[k + 1][k] = normP(w, xkrylovlen);
      }
      scale1(w, (1.0 / H[k + 1][k]), xkrylovlen);

      putColumn(V, w, k + 1, xkrylovlen);

      if (0 < k) {
        for (int j = 0; j < k; j++)
          ApplyPlaneRotation(&H[j + 1][k], &H[j][k], cs[j], sn[j]);

        getColumn(y, H, k, m + 1);
      }

      mu = sqrt(H[k][k] * H[k][k] + H[k + 1][k] * H[k + 1][k]);
      cs[k] = H[k][k] / mu;
      sn[k] = -H[k + 1][k] / mu;
      H[k][k] = cs[k] * H[k][k] - sn[k] * H[k + 1][k];
      H[k + 1][k] = 0.0;

      ApplyPlaneRotation(&s[k + 1], &s[k], cs[k], sn[k]);
      initial_error = fabs(s[k]);
      k++;
    }

    k--;
    y[k] = s[k] / H[k][k];

    for (int i = k - 1; i >= 0; i--) {
      tmp = 0.0;
      for (int l = i + 1; l <= k; l++) tmp += H[i][l] * y[l];
      y[i] = (s[i] - tmp) / H[i][i];
    }

    for (int jj = 0; jj < xkrylovlen; jj++) {
      tmp = 0.0;
      for (int l = 0; l < k; l++) tmp += y[l] * V[jj][l];
      xkrylov[jj] += tmp;
    }

    if (initial_error <= rho_tol) {
      printf("GMRES converged at restart # %d; iteration # %d  with error: %f\n", itr, k, initial_error / rho_tol * tol);

      // deallocate
      delArr(1, r);
      delArr(1, im);
      delArr(1, s);
      delArr(1, v);
      delArr(1, cs);
      delArr(1, sn);
      delArr(1, w);
      delArr(1, y);
      delArr(2, H);
      delArr(2, V);

      return;
    }
    if (GMRESVERBOSE)
      printf("Restart: %d  error: %f\n", itr, initial_error / rho_tol * tol);
  }

  printf("GMRES not converged !! Final error: %f\n", initial_error / rho_tol * tol);

  delArr(1, r);
  delArr(1, im);
  delArr(1, s);
  delArr(1, v);
  delArr(1, cs);
  delArr(1, sn);
  delArr(1, w);
  delArr(1, y);
  delArr(2, H);
  delArr(2, V);

  return;
}

void ApplyPlaneRotation(FPfield *dx, FPfield *dy, FPfield cs, FPfield sn) {
  FPfield temp = cs * *dx + sn * *dy;
  *dy = -sn * *dx +cs * *dy;
  *dx = temp;
}
