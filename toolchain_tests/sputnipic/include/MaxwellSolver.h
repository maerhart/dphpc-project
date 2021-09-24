#ifndef MAXWELLSOLVER
#define MAXWELLSOLVER

#include "Alloc.h"
#include "Basic.h"
#include "EMfield.h"
#include "InterpDensNet.h"
#include "InterpDensSpecies.h"
#include "InterpDens_aux.h"
#include "PrecisionTypes.h"
#include "Solvers.h"
#include "Smoothing.h"

void MaxwellImage(FPfield *im, 
                FPfield *vector, 
                struct EMfield *field,
                struct interpDensSpecies *ids, 
                struct grid *grd, 
                struct parameters *param);

void MaxwellSource(FPfield *bkrylov, 
                struct grid *grd, 
                struct EMfield *field,
                struct EMfield_aux *field_aux, 
                struct interpDens_aux *id_aux,
                struct parameters *param);

/** calculate the electric field using second order curl-curl formulation of
 * Maxwell equations */
void calculateE(struct grid *grd, 
                struct EMfield_aux *field_aux, 
                struct EMfield *field,
                struct interpDens_aux *id_aux, 
                struct interpDensSpecies *ids,
                struct parameters *param);

// calculate the magnetic field from Faraday's law
void calculateB(struct grid *grd, 
                struct EMfield_aux *field_aux, 
                struct EMfield *field,
                struct parameters *param);

/* Poisson Image */
void PoissonImage(FPfield *image, 
                FPfield *vector, 
                struct grid *grd);

/** calculate Poisson Correction */
void divergenceCleaning(struct grid *grd, 
                struct EMfield_aux *field_aux, 
                struct EMfield *field,
                struct interpDensNet *idn, 
                struct parameters *param);

/** Calculate hat densities: Jh and rhoh*/
void calculateHatDensities(struct interpDens_aux *id_aux,
                           struct interpDensNet *idn,
                           struct interpDensSpecies *ids, 
                           struct EMfield *field,
                           struct grid *grd, 
                           struct parameters *param);

#endif
