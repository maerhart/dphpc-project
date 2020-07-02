#ifndef BC_H
#define BC_H

#include "Grid.h"
#include "InterpDensSpecies.h"
#include "Parameters.h"
#include "PrecisionTypes.h"

/** Put Boundary Conditions on Boundaries */

//////////
// POPULATE GHOST CELL ON NODES
//////////

/** Apply BC to scalar interp quantity defined on nodes - Interpolation quantity
 */
void applyBCscalarDensN(FPinterp ***scalarN, struct grid *grd, struct parameters *param);

/** Apply BC to scalar interp quantity defined on nodes - Interpolation quantity
 */
void applyBCscalarFieldN(FPfield ***scalarN, struct grid *grd, struct parameters *param);

///////// USE THIS TO IMPOSE BC TO ELECTRIC FIELD
///////// NOW THIS IS FIXED TO ZERO

/** Apply BC to scalar interp quantity defined on nodes - Interpolation quantity
 */
void applyBCscalarFieldNzero(FPfield ***scalarN, struct grid *grd, struct parameters *param);

///////////////
////
////    add Densities
////
////
///////////////

// apply boundary conditions to species interpolated densities
void applyBCids(struct interpDensSpecies *ids, struct grid *grd,
                struct parameters *param);

//////////
// POPULATE GHOST CELL ON CELL CENTERS
//////////

/** Apply BC to scalar interp quantity defined on center- Interpolation quantity
 */
void applyBCscalarDensC(FPinterp ***scalarC, struct grid *grd, struct parameters *param);

/** Apply BC to scalar field quantity defined on center - Interpolation quantity
 */
void applyBCscalarFieldC(FPfield ***scalarC, struct grid *grd, struct parameters *param);

/** Apply BC to scalar field quantity defined on nodes - Interpolation quantity
 */
// set to zero ghost cell
void applyBCscalarFieldCzero(FPfield ***scalarC, struct grid *grd, struct parameters *param);

#endif
