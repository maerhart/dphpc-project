#include "Parameters.h"

#include <stdio.h>
#include <string.h>

void hardcodedDefaultParameters(struct parameters *param){

  /** light speed */
  param->c = 1;
  /** 4  pi */
  param->fourpi = 12.5663706144;
  /** time step */
  param->dt = 0.25;;
  /** decentering parameter */
  param->th = 1;

  /** number of time cycles */
  param->ncycles = 100;
  /** mover predictor correcto iteration */
  param->NiterMover = 3;
  /** number of particle of subcycles in the mover */
  param->n_sub_cycles = 1;

  /** number of particle batches per species when run on GPU **/
  param->number_of_batches = 16;
  /** number of threads per block to use when running kernels on GPU **/
  param->threads_per_block = 256;

  /** simulation box length - X direction   */
  param->Lx = 20;
  /** simulation box length - Y direction   */
  param->Ly = 10;
  /** simulation box length - Z direction   */
  param->Lz = 10;
  /** number of cells - X direction        */
  param->nxc = 128;
  /** number of cells - Y direction        */
  param->nyc = 64;
  /** number of cells - Z direction        */
  param->nzc = 64;
  /** object center X, e.g. planet or comet   */
  param->x_center = 0.5;
  /** object center Y, e.g. planet or comet   */
  param->y_center = 0.5;
  /** object center Z, e.g. planet or comet   */
  param->z_center = 0.5;
  /** object size - assuming a cubic box or sphere  */
  param->L_square = 0.2;

  // THIS IS IMPORTANT
  /** number of actual species */
  param->ns = 4;
  ////
  ////

  /** read input parameters for particles */

  // We read maximum 6 species from inputfile
//   array_int npcelx0 = config.read<array_int>("npcelx");
//   array_int npcely0 = config.read<array_int>("npcely");
//   array_int npcelz0 = config.read<array_int>("npcelz");
//   array_double qom0 = config.read<array_double>("qom");
//   array_double uth0 = config.read<array_double>("uth");
//   array_double vth0 = config.read<array_double>("vth");
//   array_double wth0 = config.read<array_double>("wth");
//   array_double u00 = config.read<array_double>("u0");
//   array_double v00 = config.read<array_double>("v0");
//   array_double w00 = config.read<array_double>("w0");

  param->npcelx[0] = 3;
  param->npcely[0] = 3;
  param->npcelz[0] = 3;
  param->qom[0] = -64;
  param->uth[0] = 0.045;
  param->vth[0] = 0.045;
  param->wth[0] = 0.045;
  param->u0[0] = 0;
  param->v0[0] = 0;
  param->w0[0] = 0.0065;

  if (param->ns > 1) {
    param->npcelx[1] = 3;
    param->npcely[1] = 3;
    param->npcelz[1] = 3;
    param->qom[1] = 1;
    param->uth[1] = 0.0126;
    param->vth[1] = 0.0126;
    param->wth[1] = 0.0126;
    param->u0[1] = 0;
    param->v0[1] = 0;
    param->w0[1] = -0.0325;
  }
  if (param->ns > 2) {
    param->npcelx[2] = 3;
    param->npcely[2] = 3;
    param->npcelz[2] = 3;
    param->qom[2] = -64;
    param->uth[2] = 0.045;
    param->vth[2] = 0.045;
    param->wth[2] = 0.045;
    param->u0[2] = 0;
    param->v0[2] = 0;
    param->w0[2] = 0;
  }
  if (param->ns > 3) {
    param->npcelx[3] = 3;
    param->npcely[3] = 3;
    param->npcelz[3] = 3;
    param->qom[3] = 1;
    param->uth[3] = 0.0126;
    param->vth[3] = 0.0126;
    param->wth[3] = 0.0126;
    param->u0[3] = 0;
    param->v0[3] = 0;
    param->w0[3] = 0;
  }
//   if (param->ns > 4) {
//     param->npcelx[4] = npcelx0.e;
//     param->npcely[4] = npcely0.e;
//     param->npcelz[4] = npcelz0.e;
//     param->qom[4] = qom0.e;
//     param->uth[4] = uth0.e;
//     param->vth[4] = vth0.e;
//     param->wth[4] = wth0.e;
//     param->u0[4] = u00.e;
//     param->v0[4] = v00.e;
//     param->w0[4] = w00.e;
//   }
//   if (param->ns > 5) {
//     param->npcelx[5] = npcelx0.f;
//     param->npcely[5] = npcely0.f;
//     param->npcelz[5] = npcelz0.f;
//     param->qom[5] = qom0.f;
//     param->uth[5] = uth0.f;
//     param->vth[5] = vth0.f;
//     param->wth[5] = wth0.f;
//     param->u0[5] = u00.f;
//     param->v0[5] = v00.f;
//     param->w0[5] = w00.f;
//   }

  // Initialization of densities
//   array_double rhoINIT0 = config.read<array_double>("rhoINIT");
  param->rhoINIT[0] = 1.0;
  if (param->ns > 1) param->rhoINIT[1] = 1.0;
  if (param->ns > 2) param->rhoINIT[2] = 0.1;
  if (param->ns > 3) param->rhoINIT[3] = 0.1;
//   if (param->ns > 4) param->rhoINIT[4] = rhoINIT0.e;
//   if (param->ns > 5) param->rhoINIT[5] = rhoINIT0.f;

  // Calculate the total number of particles in the domain
  param->NpMaxNpRatio = 1.0;
  int npcel = 0;
  for (int i = 0; i < param->ns; i++) {
    npcel = param->npcelx[i] * param->npcely[i] * param->npcelz[i];
    param->np[i] = npcel * param->nxc * param->nyc * param->nzc;
    param->npMax[i] = (long)(param->NpMaxNpRatio * param->np[i]);
  }

  // Boundary Conditions
  /** Periodicity for fields X **/
  param->PERIODICX = true;
  /** Periodicity for fields Y **/
  param->PERIODICY = false;
  /** Periodicity for fields Z **/
  param->PERIODICZ = true;
  /** Periodicity for Particles X **/
  param->PERIODICX_P = true;
  /** Periodicity for Particles Y **/
  param->PERIODICY_P = false;
  /** Periodicity for Particles Y **/
  param->PERIODICZ_P = true;

  // PHI Electrostatic Potential
  param->bcPHIfaceXright = 1;
  param->bcPHIfaceXleft = 1;
  param->bcPHIfaceYright = 1;
  param->bcPHIfaceYleft = 1;
  param->bcPHIfaceZright = 1;
  param->bcPHIfaceZleft = 1;

  // EM field boundary condition
  param->bcEMfaceXright = 1;
  param->bcEMfaceXleft = 1;
  param->bcEMfaceYright = 1;
  param->bcEMfaceYleft = 1;
  param->bcEMfaceZright = 1;
  param->bcEMfaceZleft = 1;

  // Particles Boundary condition
  param->bcPfaceXright = 1;
  param->bcPfaceXleft = 1;
  param->bcPfaceYright = 1;
  param->bcPfaceYleft = 1;
  param->bcPfaceZright = 1;
  param->bcPfaceZleft = 1;

  // take the injection of the particless
  param->Vinj = 0.0;

  // Initialization
  param->B0x = 0.0195;
  param->B0y = 0.0;
  param->B0z = 0.0;
  param->delta = 0.5;

  /** Smoothing quantities */
  param->SmoothON = true;  // Smoothing is ON by default
  /** Smoothing value*/
  param->SmoothValue = 0.5;  // between 0 and 1, typically 0.5
  /** Ntimes: smoothing is applied */
  param->SmoothTimes = 6;

  // Waves info
  param->Nwaves = 1;
  param->dBoB0 = 0.0;
  strcpy(param->WaveFile, "WaveFile.txt");
  param->energy = 0.018199864696222;
  param->pitch_angle = 0.698131700797732;  // 40 degree default

  param->verbose = true;

  // Poisson Correction
  param->PoissonCorrection = true;
  param->CGtol = 1E-3;
  param->GMREStol = 1E-3;

  // needed for restart (in this case no restart)
  param->first_cycle_n = 1;

  // take the output cycles
  param->FieldOutputCycle = 10;
  param->ParticlesOutputCycle = 100;
  param->RestartOutputCycle = 10000;
  param->DiagnosticsOutputCycle = 10;

  strcpy(param->SaveDirName, "data");
  strcpy(param->RestartDirName, "data");
//   param->SaveDirName = config.read<string>("SaveDirName");
//   param->RestartDirName = config.read<string>("RestartDirName");

}

/** Print Simulation Parameters */
void printParameters(struct parameters *param) {
  printf("\n-------------------------\n");
  printf("sputniPIC Sim. Parameters\n");
  printf("-------------------------\n");
  printf("Number of species    = %d\n", param->ns);
  for (int i = 0; i < param->ns; i++)
    printf("Number of particles of species %d = %ld\t (MAX = %ld) QOM = %f\n", 
      i, param->np[i], param->npMax[i], param->qom[i]);

  printf("x-Length                 = %f\n", param->Lx);
  printf("y-Length                 = %f\n", param->Ly);
  printf("z-Length                 = %f\n", param->Lz);
  printf("Number of cells (x)      = %d\n", param->nxc);
  printf("Number of cells (y)      = %d\n", param->nyc);
  printf("Number of cells (z)      = %d\n", param->nzc);
  printf("Time step                = %f\n", param->dt);
  printf("Number of cycles         = %d\n", param->ncycles);
  printf("Results saved in: %s\n", param->SaveDirName);
}