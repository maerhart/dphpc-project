
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "IO.h"



void _vtk_write_vector(
    FILE *file, char *field, 
    int nxn, int nyn, int nzn, 
    double dx, double dy, double dz,
    FPfield*** X, FPfield*** Y, FPfield*** Z){

  fprintf(file, "# vtk DataFile Version 1.0\n");
  fprintf(file, "%s field\n", field);
  fprintf(file, "ASCII\n");
  fprintf(file, "DATASET STRUCTURED_POINTS\n");
  fprintf(file, "DIMENSIONS %d %d %d\n", (nxn - 3), (nyn - 3), (nzn - 3));
  fprintf(file, "ORIGIN %f %f %f\n", 0.0, 0.0, 0.0);
  fprintf(file, "SPACING %.10f %.10f %.10f\n", dx, dy, dz);
  fprintf(file, "POINT_DATA %d\n", (nxn - 3) * (nyn - 3) * (nzn - 3));
  fprintf(file, "VECTORS %s float\n", field);

  double Ex = 0, Ey = 0, Ez = 0;

  for (int k = 1; k < nzn - 2; k++)
    for (int j = 1; j < nyn - 2; j++)
      for (int i = 1; i < nxn - 2; i++) {
        Ex = X[i][j][k];
        if (fabs(Ex) < 1E-8) Ex = 0.0;
        Ey = Y[i][j][k];
        if (fabs(Ey) < 1E-8) Ey = 0.0;
        Ez = Z[i][j][k];
        if (fabs(Ez) < 1E-8) Ez = 0.0;
        fprintf(file, "%.10f %.10f %.10f\n", Ex, Ey, Ez);
      }
}


void _vtk_write_scalar(
    FILE *file,
    char *name, char *id,
    int nxn, int nyn, int nzn,
    double dx, double dy, double dz,
    FPinterp ***data
    ){

  fprintf(file, "# vtk DataFile Version 1.0\n");
  fprintf(file, "%s \n", name);
  fprintf(file, "ASCII\n");
  fprintf(file, "DATASET STRUCTURED_POINTS\n");
  fprintf(file, "DIMENSIONS %d %d %d\n", (nxn - 3), (nyn - 3), (nzn - 3));
  fprintf(file, "ORIGIN %f %f %f\n", 0.0, 0.0, 0.0);
  fprintf(file, "SPACING %.10f %.10f %.10f\n", dx, dy, dz);
  fprintf(file, "POINT_DATA %d\n", (nxn - 3) * (nyn - 3) * (nzn - 3));
  fprintf(file, "SCALARS %s float\n", id);
  fprintf(file, "LOOKUP_TABLE default\n");

  for (int k = 1; k < nzn - 2; k++)
    for (int j = 1; j < nyn - 2; j++)
      for (int i = 1; i < nxn - 2; i++) {
        fprintf(file, "%.10f\n", data[i][j][k]);
      }
}


void VTK_Write_Vectors(int cycle, struct grid *grd, struct EMfield *field, struct parameters *param) {
  // stream file to be opened and managed

  char filename[MAX_STRING_LEN+20];
  FILE *file;

  // ========================================================================//
  // E field
  sprintf(filename, "%s/E_%d.vtk", param->SaveDirName, cycle);

  printf("Opening file: %s\n", filename);
  file = fopen(filename, "w");

  _vtk_write_vector(
    file, "E", 
    grd->nxn, grd->nyn, grd->nzn,
    grd->dx, grd->dy, grd->dz,
    field->Ex, field->Ey, field->Ez
    );

  fclose(file);

  // ========================================================================//
  // B field
  sprintf(filename, "%s/B_%d.vtk", param->SaveDirName, cycle);

  printf("Opening file: %s\n", filename);
  file = fopen(filename, "w");

  _vtk_write_vector(
    file, "E", 
    grd->nxn, grd->nyn, grd->nzn,
    grd->dx, grd->dy, grd->dz,
    field->Bxn, field->Byn, field->Bzn
    );

    fclose(file);
}

void VTK_Write_Scalars(int cycle, struct grid *grd,
                       struct interpDensSpecies *ids,
                       struct interpDensNet *idn,
                       struct parameters *param) {
  // stream file to be opened and managed


  char filename[MAX_STRING_LEN+12];
  FILE *file;

  // ========================================================================//
  // rho_e
  sprintf(filename, "%s/rhoe_%d.vtk", param->SaveDirName, cycle);
  printf("Opening file: %s\n", filename);
  file = fopen(filename, "w");

  _vtk_write_scalar(
    file,
    "Electron Density - Current Sheet", "rhoe",
    grd->nxn, grd->nyn, grd->nzn,
    grd->dx, grd->dy, grd->dz,
    ids[0].rhon
  );

  fclose(file);

  // ========================================================================//
  // rho_i
  sprintf(filename, "%s/rhoi_%d.vtk", param->SaveDirName, cycle);
  printf("Opening file: %s\n", filename);
  file = fopen(filename, "w");

  _vtk_write_scalar(
    file,
    "Ion Density - Current Sheet", "rhoi",
    grd->nxn, grd->nyn, grd->nzn,
    grd->dx, grd->dy, grd->dz,
    ids[1].rhon
  );

  fclose(file);

  // ========================================================================//
  // rho_net
  sprintf(filename, "%s/rho_net_%d.vtk", param->SaveDirName, cycle);
  printf("Opening file: %s\n", filename);
  file = fopen(filename, "w");

  _vtk_write_scalar(
    file,
    "Net Charge Density", "rhonet",
    grd->nxn, grd->nyn, grd->nzn,
    grd->dx, grd->dy, grd->dz,
    idn->rhon
  );

  fclose(file);

}
