#ifndef SOLVERS_H
#define SOLVERS_H

#include <math.h>
#include "EMfield.h"
#include "Grid.h"
#include "InterpDensSpecies.h"
#include "Parameters.h"
#include "PrecisionTypes.h"

/* CG */
typedef void (*GENERIC_IMAGE)(FPfield *, FPfield *, struct grid *);
bool CG(FPfield *xkrylov, int xkrylovlen, FPfield *b, int maxit, double tol,
        GENERIC_IMAGE FunctionImage, struct grid *grd);

/* GMRES */
typedef void (*GENERIC_IMAGE_GMRES)(FPfield *, FPfield *, struct EMfield *,
                                    struct interpDensSpecies *, struct grid *, struct parameters *);
void ApplyPlaneRotation(FPfield *dx, FPfield *dy, FPfield cs, FPfield sn);
void GMRes(GENERIC_IMAGE_GMRES FunctionImage, FPfield *xkrylov, int xkrylovlen,
           FPfield *b, int m, int max_iter, double tol, struct EMfield *field,
           struct interpDensSpecies *ids, struct grid *grd, struct parameters *param);

#endif
