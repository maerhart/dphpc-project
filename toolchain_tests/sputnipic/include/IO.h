#ifndef IO_H
#define IO_H

#include "EMfield.h"
#include "Grid.h"
#include "InterpDensNet.h"
#include "InterpDensSpecies.h"
#include "Parameters.h"


void VTK_Write_Vectors(int cycle, 
                        struct grid *grd, 
                        struct EMfield *field, 
                        struct parameters *param);

void VTK_Write_Scalars(int cycle, 
                        struct grid *grd,
                        struct interpDensSpecies *ids,
                        struct interpDensNet *idn,
                        struct parameters *param);

#endif
