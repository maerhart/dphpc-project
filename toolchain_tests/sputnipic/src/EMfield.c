#include "EMfield.h"

/** allocate electric and magnetic field */
void field_allocate(struct grid *grd, struct EMfield *field) {

  // E on nodes
  field->Ex = (FPfield***) ptrArr((void**) &field->Ex_flat, sizeof(FPfield), 3, grd->nxn, grd->nyn, grd->nzn);
  field->Ey = (FPfield***) ptrArr((void**) &field->Ey_flat, sizeof(FPfield), 3, grd->nxn, grd->nyn, grd->nzn);
  field->Ez = (FPfield***) ptrArr((void**) &field->Ez_flat, sizeof(FPfield), 3, grd->nxn, grd->nyn, grd->nzn);

  // B on nodes
  field->Bxn = (FPfield***) ptrArr((void**) &field->Bxn_flat, sizeof(FPfield), 3, grd->nxn, grd->nyn, grd->nzn);
  field->Byn = (FPfield***) ptrArr((void**) &field->Byn_flat, sizeof(FPfield), 3, grd->nxn, grd->nyn, grd->nzn);
  field->Bzn = (FPfield***) ptrArr((void**) &field->Bzn_flat, sizeof(FPfield), 3, grd->nxn, grd->nyn, grd->nzn);

}

/** deallocate electric and magnetic field */
void field_deallocate(struct grid *grd, struct EMfield *field) {

  delArr(3, (void*) field->Ex);
  delArr(3, (void*) field->Ey);
  delArr(3, (void*) field->Ez);

  delArr(3, (void*) field->Bxn);
  delArr(3, (void*) field->Byn);
  delArr(3, (void*) field->Bzn);

}

/** allocate electric and magnetic field */
void field_aux_allocate(struct grid *grd, struct EMfield_aux *field_aux) {
  // Electrostatic potential
  field_aux->Phi =
      (FPfield***) ptrArr((void**) &field_aux->Phi_flat, sizeof(FPfield), 3, grd->nxc, grd->nyc, grd->nzc);

  // allocate 3D arrays
  field_aux->Exth =
      (FPfield***) ptrArr((void**) &field_aux->Exth_flat, sizeof(FPfield), 3, grd->nxn, grd->nyn, grd->nzn);
  field_aux->Eyth =
      (FPfield***) ptrArr((void**) &field_aux->Eyth_flat, sizeof(FPfield), 3, grd->nxn, grd->nyn, grd->nzn);
  field_aux->Ezth =
      (FPfield***) ptrArr((void**) &field_aux->Ezth_flat, sizeof(FPfield), 3, grd->nxn, grd->nyn, grd->nzn);
  // B on centers
  field_aux->Bxc =
      (FPfield***) ptrArr((void**) &field_aux->Bxc_flat, sizeof(FPfield), 3, grd->nxc, grd->nyc, grd->nzc);
  field_aux->Byc =
      (FPfield***) ptrArr((void**) &field_aux->Byc_flat, sizeof(FPfield), 3, grd->nxc, grd->nyc, grd->nzc);
  field_aux->Bzc =
      (FPfield***) ptrArr((void**) &field_aux->Bzc_flat, sizeof(FPfield), 3, grd->nxc, grd->nyc, grd->nzc);
}

/** deallocate */
void field_aux_deallocate(struct grid *grd, struct EMfield_aux *field_aux) {
  // Eth
  delArr(3, (void*) field_aux->Exth);
  delArr(3, (void*) field_aux->Eyth);
  delArr(3, (void*) field_aux->Ezth);
  // Bc
  delArr(3, (void*) field_aux->Bxc);
  delArr(3, (void*) field_aux->Byc);
  delArr(3, (void*) field_aux->Bzc);
  // Phi
  delArr(3, (void*) field_aux->Phi);
}
