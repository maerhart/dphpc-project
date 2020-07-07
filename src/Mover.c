#include "Mover.h"
#include "mpi_comm.h"


int batch_update_particles(
	struct particles *part_global, 
	struct particles *part_local,
	struct EMfield *field,
	struct interpDensSpecies *ids,
	struct grid *grd,
	struct parameters *param,
	int batchsize,
	long tot_num_particles
	){

	int offset = 0;
	int np;

	int batches = 0;

	while (offset < tot_num_particles) {

		np = mpi_scatter_particles(part_global, part_local, offset, batchsize, tot_num_particles);

		mover_PC(part_local, field, grd, param);
		interpP2G(part_local, ids, grd);

		mpi_gather_particles(part_global, part_local, offset, batchsize, tot_num_particles);

		offset += np;
		batches++;
	}

	return batches;

}


/** particle mover  using predictor-corrector */
int mover_PC(
	struct particles *part, 
	struct EMfield *field, 
	struct grid *grd,
	struct parameters *param) {
	// print species and subcycling
	// std::cout << "***  MOVER  ITERATIONS = " << part->NiterMover << " - Species "
	//           << part->species_ID << " ***" << std::endl;

	// auxiliary variables
	FPpart dt_sub_cycling = (FPpart)param->dt / ((double)part->n_sub_cycles);
	FPpart dto2 = .5 * dt_sub_cycling, qomdt2 = part->qom * dto2 / param->c;
	FPpart omdtsq, denom, ut, vt, wt, udotb;

	// local (to the particle) electric and magnetic field
	FPfield Exl = 0.0, Eyl = 0.0, Ezl = 0.0, Bxl = 0.0, Byl = 0.0, Bzl = 0.0;

	// interpolation densities
	int ix, iy, iz;
	FPfield weight[2][2][2];
	FPfield xi[2], eta[2], zeta[2];

	// intermediate particle position and velocity
	FPpart xptilde, yptilde, zptilde, uptilde, vptilde, wptilde;

// move each particle with new fields
#pragma omp parallel for private(xptilde, yptilde, zptilde, uptilde, vptilde,  \
		wptilde, Exl, Eyl, Ezl, Bxl, Byl, Bzl,        \
		weight, xi, eta, zeta, omdtsq, denom, ut, vt, \
		wt, udotb, ix, iy, iz)
	for (int i = 0; i < part->nop; i++) {

		xptilde = part->x[i];
		yptilde = part->y[i];
		zptilde = part->z[i];
		// calculate the average velocity iteratively
		for (int innter = 0; innter < part->NiterMover; innter++) {
			// interpolation G-->P
			ix = 2 + (part->x[i] - grd->xStart) * grd->invdx;
			iy = 2 + (part->y[i] - grd->yStart) * grd->invdy;
			iz = 2 + (part->z[i] - grd->zStart) * grd->invdz;

			// calculate weights
			xi[0] = part->x[i] - grd->XN[ix - 1][iy][iz];
			eta[0] = part->y[i] - grd->YN[ix][iy - 1][iz];
			zeta[0] = part->z[i] - grd->ZN[ix][iy][iz - 1];
			xi[1] = grd->XN[ix][iy][iz] - part->x[i];
			eta[1] = grd->YN[ix][iy][iz] - part->y[i];
			zeta[1] = grd->ZN[ix][iy][iz] - part->z[i];
			for (int ii = 0; ii < 2; ii++)
				for (int jj = 0; jj < 2; jj++)
					for (int kk = 0; kk < 2; kk++)
						weight[ii][jj][kk] = xi[ii] * eta[jj] * zeta[kk] * grd->invVOL;

			// set to zero local electric and magnetic field
			Exl = 0.0, Eyl = 0.0, Ezl = 0.0, Bxl = 0.0, Byl = 0.0, Bzl = 0.0;

			for (int ii = 0; ii < 2; ii++)
				for (int jj = 0; jj < 2; jj++)
					for (int kk = 0; kk < 2; kk++) {
						Exl += weight[ii][jj][kk] * field->Ex[ix - ii][iy - jj][iz - kk];
						Eyl += weight[ii][jj][kk] * field->Ey[ix - ii][iy - jj][iz - kk];
						Ezl += weight[ii][jj][kk] * field->Ez[ix - ii][iy - jj][iz - kk];
						Bxl += weight[ii][jj][kk] * field->Bxn[ix - ii][iy - jj][iz - kk];
						Byl += weight[ii][jj][kk] * field->Byn[ix - ii][iy - jj][iz - kk];
						Bzl += weight[ii][jj][kk] * field->Bzn[ix - ii][iy - jj][iz - kk];
					}

			// end interpolation
			omdtsq = qomdt2 * qomdt2 * (Bxl * Bxl + Byl * Byl + Bzl * Bzl);
			denom = 1.0 / (1.0 + omdtsq);
			// solve the position equation
			ut = part->u[i] + qomdt2 * Exl;
			vt = part->v[i] + qomdt2 * Eyl;
			wt = part->w[i] + qomdt2 * Ezl;
			udotb = ut * Bxl + vt * Byl + wt * Bzl;
			// solve the velocity equation
			uptilde =
					(ut + qomdt2 * (vt * Bzl - wt * Byl + qomdt2 * udotb * Bxl)) * denom;
			vptilde =
					(vt + qomdt2 * (wt * Bxl - ut * Bzl + qomdt2 * udotb * Byl)) * denom;
			wptilde =
					(wt + qomdt2 * (ut * Byl - vt * Bxl + qomdt2 * udotb * Bzl)) * denom;
			// update position
			part->x[i] = xptilde + uptilde * dto2;
			part->y[i] = yptilde + vptilde * dto2;
			part->z[i] = zptilde + wptilde * dto2;

		}  // end of iteration
		// update the final position and velocity
		part->u[i] = 2.0 * uptilde - part->u[i];
		part->v[i] = 2.0 * vptilde - part->v[i];
		part->w[i] = 2.0 * wptilde - part->w[i];
		part->x[i] = xptilde + uptilde * dt_sub_cycling;
		part->y[i] = yptilde + vptilde * dt_sub_cycling;
		part->z[i] = zptilde + wptilde * dt_sub_cycling;

		//////////
		//////////
		////////// BC

		// X-DIRECTION: BC particles
		if (part->x[i] > grd->Lx) {
			if (param->PERIODICX == true) {  // PERIODIC
				part->x[i] = part->x[i] - grd->Lx;
			} else {  // REFLECTING BC
				part->u[i] = -part->u[i];
				part->x[i] = 2 * grd->Lx - part->x[i];
			}
		}

		if (part->x[i] < 0) {
			if (param->PERIODICX == true) {  // PERIODIC
				part->x[i] = part->x[i] + grd->Lx;
			} else {  // REFLECTING BC
				part->u[i] = -part->u[i];
				part->x[i] = -part->x[i];
			}
		}

		// Y-DIRECTION: BC particles
		if (part->y[i] > grd->Ly) {
			if (param->PERIODICY == true) {  // PERIODIC
				part->y[i] = part->y[i] - grd->Ly;
			} else {  // REFLECTING BC
				part->v[i] = -part->v[i];
				part->y[i] = 2 * grd->Ly - part->y[i];
			}
		}

		if (part->y[i] < 0) {
			if (param->PERIODICY == true) {  // PERIODIC
				part->y[i] = part->y[i] + grd->Ly;
			} else {  // REFLECTING BC
				part->v[i] = -part->v[i];
				part->y[i] = -part->y[i];
			}
		}

		// Z-DIRECTION: BC particles
		if (part->z[i] > grd->Lz) {
			if (param->PERIODICZ == true) {  // PERIODIC
				part->z[i] = part->z[i] - grd->Lz;
			} else {  // REFLECTING BC
				part->w[i] = -part->w[i];
				part->z[i] = 2 * grd->Lz - part->z[i];
			}
		}

		if (part->z[i] < 0) {
			if (param->PERIODICZ == true) {  // PERIODIC
				part->z[i] = part->z[i] + grd->Lz;
			} else {  // REFLECTING BC
				part->w[i] = -part->w[i];
				part->z[i] = -part->z[i];
			}
		}

	}  // end of particles

	return (0);  // exit succcesfully
}  // end of the mover

/** particle mover  using predictor-corrector (formulated to be automatically
 * vectorized  */
int mover_PC_V(struct particles *part, struct EMfield *field, struct grid *grd,
							 struct parameters *param) {
	// print species and subcycling
	printf("***  MOVER  ITERATIONS = %d - Species %d  ***\n", part->NiterMover, part->species_ID);

	// auxiliary variables
	FPpart dt_sub_cycling = (FPpart)param->dt / ((double)part->n_sub_cycles);
	FPpart dto2 = .5 * dt_sub_cycling, qomdt2 = part->qom * dto2 / param->c;
	FPpart omdtsq, denom, ut, vt, wt, udotb;
	// FPpart vel;

	// interpolation densities
	int ix, iy, iz;
	FPfield weight[2][2][2];
	FPfield xi[2], eta[2], zeta[2];

	int num_part = part->nop;

	// intermediate particle position and velocity

	FPpart *xptilde = (FPpart*) newArr(sizeof(FPpart), 1, num_part); // new FPpart[num_part];
	FPpart *yptilde = (FPpart*) newArr(sizeof(FPpart), 1, num_part); // new FPpart[num_part];
	FPpart *zptilde = (FPpart*) newArr(sizeof(FPpart), 1, num_part); // new FPpart[num_part];
	FPpart *uptilde = (FPpart*) newArr(sizeof(FPpart), 1, num_part); // new FPpart[num_part];
	FPpart *vptilde = (FPpart*) newArr(sizeof(FPpart), 1, num_part); // new FPpart[num_part];
	FPpart *wptilde = (FPpart*) newArr(sizeof(FPpart), 1, num_part); // new FPpart[num_part];

	FPpart *Exl = (FPpart*) newArr(sizeof(FPpart), 1, num_part); // new FPpart[num_part];
	FPpart *Eyl = (FPpart*) newArr(sizeof(FPpart), 1, num_part); // new FPpart[num_part];
	FPpart *Ezl = (FPpart*) newArr(sizeof(FPpart), 1, num_part); // new FPpart[num_part];
	FPpart *Bxl = (FPpart*) newArr(sizeof(FPpart), 1, num_part); // new FPpart[num_part];
	FPpart *Byl = (FPpart*) newArr(sizeof(FPpart), 1, num_part); // new FPpart[num_part];
	FPpart *Bzl = (FPpart*) newArr(sizeof(FPpart), 1, num_part); // new FPpart[num_part];

	for (int ip = 0; ip < num_part; ip++) {
		xptilde[ip] = part->x[ip];
		yptilde[ip] = part->y[ip];
		zptilde[ip] = part->z[ip];
		uptilde[ip] = part->u[ip];
		vptilde[ip] = part->v[ip];
		wptilde[ip] = part->w[ip];
	}

	// calculate the average velocity iteratively
	for (int innter = 0; innter < part->NiterMover; innter++) {
		for (int ip = 0; ip < num_part; ip++) {
			Exl[ip] = 0.0;
			Eyl[ip] = 0.0;
			Ezl[ip] = 0.0;
			Bxl[ip] = 0.0;
			Byl[ip] = 0.0;
			Bzl[ip] = 0.0;
		}

		// calculate array with local field
		// this part can't be vectorized trivially
		for (int ip = 0; ip < num_part; ip++) {
			// interpolation G-->P
			ix = 2 + (int)((part->x[ip] - grd->xStart) * grd->invdx);
			iy = 2 + (int)((part->y[ip] - grd->yStart) * grd->invdy);
			iz = 2 + (int)((part->z[ip] - grd->zStart) * grd->invdz);

			// calculate weights
			xi[0] = part->x[ip] - grd->XN[ix - 1][iy][iz];
			eta[0] = part->y[ip] - grd->YN[ix][iy - 1][iz];
			zeta[0] = part->z[ip] - grd->ZN[ix][iy][iz - 1];
			xi[1] = grd->XN[ix][iy][iz] - part->x[ip];
			eta[1] = grd->YN[ix][iy][iz] - part->y[ip];
			zeta[1] = grd->ZN[ix][iy][iz] - part->z[ip];
			for (int ii = 0; ii < 2; ii++)
				for (int jj = 0; jj < 2; jj++)
					for (int kk = 0; kk < 2; kk++)
						weight[ii][jj][kk] = xi[ii] * eta[jj] * zeta[kk] * grd->invVOL;

			for (int ii = 0; ii < 2; ii++)
				for (int jj = 0; jj < 2; jj++)
					for (int kk = 0; kk < 2; kk++) {
						Exl[ip] +=
								weight[ii][jj][kk] * field->Ex[ix - ii][iy - jj][iz - kk];
						Eyl[ip] +=
								weight[ii][jj][kk] * field->Ey[ix - ii][iy - jj][iz - kk];
						Ezl[ip] +=
								weight[ii][jj][kk] * field->Ez[ix - ii][iy - jj][iz - kk];
						Bxl[ip] +=
								weight[ii][jj][kk] * field->Bxn[ix - ii][iy - jj][iz - kk];
						Byl[ip] +=
								weight[ii][jj][kk] * field->Byn[ix - ii][iy - jj][iz - kk];
						Bzl[ip] +=
								weight[ii][jj][kk] * field->Bzn[ix - ii][iy - jj][iz - kk];
					}

		}  // end of np

// It is important to vectorize this
		for (int i = 0; i < num_part; i++) {
			omdtsq = qomdt2 * qomdt2 *
							 (Bxl[i] * Bxl[i] + Byl[i] * Byl[i] + Bzl[i] * Bzl[i]);
			denom = 1.0 / (1.0 + omdtsq);
			// solve the position equation
			ut = part->u[i] + qomdt2 * Exl[i];
			vt = part->v[i] + qomdt2 * Eyl[i];
			wt = part->w[i] + qomdt2 * Ezl[i];
			udotb = ut * Bxl[i] + vt * Byl[i] + wt * Bzl[i];
			// solve the velocity equation
			uptilde[i] = (ut + qomdt2 * (vt * Bzl[i] - wt * Byl[i] +
																	 qomdt2 * udotb * Bxl[i])) *
									 denom;
			vptilde[i] = (vt + qomdt2 * (wt * Bxl[i] - ut * Bzl[i] +
																	 qomdt2 * udotb * Byl[i])) *
									 denom;
			wptilde[i] = (wt + qomdt2 * (ut * Byl[i] - vt * Bxl[i] +
																	 qomdt2 * udotb * Bzl[i])) *
									 denom;

			// update position
			part->x[i] = xptilde[i] + uptilde[i] * dto2;
			part->y[i] = yptilde[i] + vptilde[i] * dto2;
			part->z[i] = zptilde[i] + wptilde[i] * dto2;
		}

	}  // end of iteration of predictor corrector

	for (int i = 0; i < num_part; i++) {
		// update the final position and velocity
		part->u[i] = 2.0 * uptilde[i] - part->u[i];
		part->v[i] = 2.0 * vptilde[i] - part->v[i];
		part->w[i] = 2.0 * wptilde[i] - part->w[i];
		part->x[i] = xptilde[i] + uptilde[i] * dt_sub_cycling;
		part->y[i] = yptilde[i] + vptilde[i] * dt_sub_cycling;
		part->z[i] = zptilde[i] + wptilde[i] * dt_sub_cycling;
	}

	// BC: Boundary Conditions
	// X - DIRECTION
	if (param->PERIODICX == true) {
		for (int i = 0; i < num_part; i++) {
			if (part->x[i] > grd->Lx) part->x[i] = part->x[i] - grd->Lx;
			if (part->x[i] < 0) part->x[i] = part->x[i] + grd->Lx;
		}
	} else {  // perfect conductor
		for (int i = 0; i < num_part; i++) {
			if (part->x[i] > grd->Lx) part->x[i] = 2 * grd->Lx - part->x[i];
			if (part->x[i] < 0) part->x[i] = -part->x[i];
		}  // end of loop

		// this is not vectorized by CLANG
		for (int i = 0; i < num_part; i++) {
			if (part->x[i] > grd->Lx) part->u[i] = -part->u[i];
			if (part->x[i] < 0) part->u[i] = -part->u[i];
		}  // end of loop
	}

	// Y - DIRECTION
	if (param->PERIODICY == true) {
		for (int i = 0; i < num_part; i++) {
			if (part->y[i] > grd->Ly) part->y[i] = part->y[i] - grd->Ly;
			if (part->y[i] < 0) part->y[i] = part->y[i] + grd->Ly;
		}
	} else {  // perfect conductor
		for (int i = 0; i < num_part; i++) {
			if (part->y[i] > grd->Ly) part->y[i] = 2 * grd->Ly - part->y[i];
			if (part->y[i] < 0) part->y[i] = -part->y[i];
		}  // end of loop

		// this is not vectorized by CLANG
		for (int i = 0; i < num_part; i++) {
			if (part->y[i] > grd->Ly) part->v[i] = -part->v[i];
			if (part->y[i] < 0) part->v[i] = -part->v[i];
		}  // end of loop
	}

	// Z - DIRECTION
	if (param->PERIODICZ == true) {
		for (int i = 0; i < num_part; i++) {
			if (part->z[i] > grd->Lz) part->z[i] = part->z[i] - grd->Lz;
			if (part->z[i] < 0) part->z[i] = part->z[i] + grd->Lz;
		}
	} else {  // perfect conductor
		for (int i = 0; i < num_part; i++) {
			if (part->z[i] > grd->Lz) part->z[i] = 2 * grd->Lz - part->z[i];
			if (part->z[i] < 0) part->z[i] = -part->z[i];
		}  // end of loop

		// this is not vectorized by CLANG
		for (int i = 0; i < num_part; i++) {
			if (part->z[i] > grd->Lz) part->w[i] = -part->w[i];
			if (part->z[i] < 0) part->w[i] = -part->w[i];
		}  // end of loop
	}

	// deallocate
	delArr(1, xptilde); // delete[] xptilde;
	delArr(1, yptilde); // delete[] yptilde;
	delArr(1, zptilde); // delete[] zptilde;
	delArr(1, uptilde); // delete[] uptilde;
	delArr(1, vptilde); // delete[] vptilde;
	delArr(1, wptilde); // delete[] wptilde;

	delArr(1, Exl); // delete[] Exl;
	delArr(1, Eyl); // delete[] Eyl;
	delArr(1, Ezl); // delete[] Ezl;
	delArr(1, Bxl); // delete[] Bxl;
	delArr(1, Byl); // delete[] Byl;
	delArr(1, Bzl); // delete[] Bzl;

	return (0);  // exit succcesfully
}  // end of the mover

/** Interpolation Particle --> Grid: This is for species */
void interpP2G(struct particles *part, struct interpDensSpecies *ids,
							 struct grid *grd) {
	// arrays needed for interpolation
	FPpart weight[2][2][2];
	FPpart temp[2][2][2];
	FPpart xi[2], eta[2], zeta[2];

	// index of the cell
	int ix, iy, iz;

	// it is cheaper to recover from a race condition than protect against it
	//#pragma omp parallel for private(temp, weight, xi, eta, zeta, ix, iy, iz)
	for (long i = 0; i < part->nop; i++) {
		// determine cell: can we change to int()? is it faster?
		ix = 2 + (int)(floor((part->x[i] - grd->xStart) * grd->invdx));
		iy = 2 + (int)(floor((part->y[i] - grd->yStart) * grd->invdy));
		iz = 2 + (int)(floor((part->z[i] - grd->zStart) * grd->invdz));

		// distances from node
		xi[0] = part->x[i] - grd->XN[ix - 1][iy][iz];
		eta[0] = part->y[i] - grd->YN[ix][iy - 1][iz];
		zeta[0] = part->z[i] - grd->ZN[ix][iy][iz - 1];
		xi[1] = grd->XN[ix][iy][iz] - part->x[i];
		eta[1] = grd->YN[ix][iy][iz] - part->y[i];
		zeta[1] = grd->ZN[ix][iy][iz] - part->z[i];


		/*
#pragma omp for collapse(3)
for (int ii = 0; ii < 2; ii++)
for (int jj = 0; jj < 2; jj++)
for (int kk = 0; kk < 2; kk++){
		weight[ii][jj][kk] = part->q[i] * xi[ii] * eta[jj] * zeta[kk] *
grd->invVOL;
		// rho
		ids->rhon[ix - ii][iy - jj][iz - kk] += weight[ii][jj][kk] *
grd->invVOL;
		// Jx
		ids->Jx[ix - ii][iy - jj][iz - kk] += part->u[i] *
weight[ii][jj][kk] * grd->invVOL;
		// Jy
		ids->Jy[ix - ii][iy - jj][iz - kk] += part->v[i] *
weight[ii][jj][kk] * grd->invVOL;
		// Jz
		ids->Jz[ix - ii][iy - jj][iz - kk] += part->w[i] *
weight[ii][jj][kk]* grd->invVOL;
		// pxx
		ids->pxx[ix - ii][iy - jj][iz - kk] +=  part->u[i] * part->u[i]
* weight[ii][jj][kk] * grd->invVOL;
		// pxy
		ids->pxy[ix - ii][iy - jj][iz - kk] +=  part->u[i] * part->v[i]
* weight[ii][jj][kk] * grd->invVOL;
		// pxz
		ids->pxz[ix - ii][iy - jj][iz - kk] +=  part->u[i] * part->w[i]
* weight[ii][jj][kk] * grd->invVOL;
		// pyy
		ids->pyy[ix - ii][iy - jj][iz - kk] +=  part->v[i] * part->v[i]
* weight[ii][jj][kk] * grd->invVOL;
		// pyz
		ids->pyz[ix - ii][iy - jj][iz - kk] +=  part->v[i] * part->w[i]
* weight[ii][jj][kk] * grd->invVOL;
		// pzz
		ids->pzz[ix - ii][iy - jj][iz - kk] +=  part->w[i] * part->w[i]
* weight[ii][jj][kk] * grd->invVOL;

}

*/

		// calculate the weights for different nodes
		for (int ii = 0; ii < 2; ii++)
			for (int jj = 0; jj < 2; jj++)
				for (int kk = 0; kk < 2; kk++){
					weight[ii][jj][kk] =
							part->q[i] * xi[ii] * eta[jj] * zeta[kk] * grd->invVOL;
							// if (i % 100000 == 0){
							//   printf("%d %ld : %d %d %d %.8f \n", 
							//     part->species_ID, i, ii, jj, kk, weight[ii][jj][kk]);
							// }
				}

		//////////////////////////
		// add charge density
		for (int ii = 0; ii < 2; ii++)
			for (int jj = 0; jj < 2; jj++)
				for (int kk = 0; kk < 2; kk++)
					ids->rhon[ix - ii][iy - jj][iz - kk] +=
							weight[ii][jj][kk] * grd->invVOL;

		////////////////////////////
		// add current density - Jx
		for (int ii = 0; ii < 2; ii++)
			for (int jj = 0; jj < 2; jj++)
				for (int kk = 0; kk < 2; kk++)
					temp[ii][jj][kk] = part->u[i] * weight[ii][jj][kk];

		for (int ii = 0; ii < 2; ii++)
			for (int jj = 0; jj < 2; jj++)
				for (int kk = 0; kk < 2; kk++)
					ids->Jx[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;

		////////////////////////////
		// add current density - Jy
		for (int ii = 0; ii < 2; ii++)
			for (int jj = 0; jj < 2; jj++)
				for (int kk = 0; kk < 2; kk++)
					temp[ii][jj][kk] = part->v[i] * weight[ii][jj][kk];
		for (int ii = 0; ii < 2; ii++)
			for (int jj = 0; jj < 2; jj++)
				for (int kk = 0; kk < 2; kk++)
					ids->Jy[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;

		////////////////////////////
		// add current density - Jz
		for (int ii = 0; ii < 2; ii++)
			for (int jj = 0; jj < 2; jj++)
				for (int kk = 0; kk < 2; kk++)
					temp[ii][jj][kk] = part->w[i] * weight[ii][jj][kk];
		for (int ii = 0; ii < 2; ii++)
			for (int jj = 0; jj < 2; jj++)
				for (int kk = 0; kk < 2; kk++)
					ids->Jz[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;

		////////////////////////////
		// add pressure pxx
		for (int ii = 0; ii < 2; ii++)
			for (int jj = 0; jj < 2; jj++)
				for (int kk = 0; kk < 2; kk++)
					temp[ii][jj][kk] = part->u[i] * part->u[i] * weight[ii][jj][kk];
		for (int ii = 0; ii < 2; ii++)
			for (int jj = 0; jj < 2; jj++)
				for (int kk = 0; kk < 2; kk++)
					ids->pxx[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;

		////////////////////////////
		// add pressure pxy
		for (int ii = 0; ii < 2; ii++)
			for (int jj = 0; jj < 2; jj++)
				for (int kk = 0; kk < 2; kk++)
					temp[ii][jj][kk] = part->u[i] * part->v[i] * weight[ii][jj][kk];
		for (int ii = 0; ii < 2; ii++)
			for (int jj = 0; jj < 2; jj++)
				for (int kk = 0; kk < 2; kk++)
					ids->pxy[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;

		/////////////////////////////
		// add pressure pxz
		for (int ii = 0; ii < 2; ii++)
			for (int jj = 0; jj < 2; jj++)
				for (int kk = 0; kk < 2; kk++)
					temp[ii][jj][kk] = part->u[i] * part->w[i] * weight[ii][jj][kk];
		for (int ii = 0; ii < 2; ii++)
			for (int jj = 0; jj < 2; jj++)
				for (int kk = 0; kk < 2; kk++)
					ids->pxz[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;

		/////////////////////////////
		// add pressure pyy
		for (int ii = 0; ii < 2; ii++)
			for (int jj = 0; jj < 2; jj++)
				for (int kk = 0; kk < 2; kk++)
					temp[ii][jj][kk] = part->v[i] * part->v[i] * weight[ii][jj][kk];
		for (int ii = 0; ii < 2; ii++)
			for (int jj = 0; jj < 2; jj++)
				for (int kk = 0; kk < 2; kk++)
					ids->pyy[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;

		/////////////////////////////
		// add pressure pyz
		for (int ii = 0; ii < 2; ii++)
			for (int jj = 0; jj < 2; jj++)
				for (int kk = 0; kk < 2; kk++)
					temp[ii][jj][kk] = part->v[i] * part->w[i] * weight[ii][jj][kk];
		for (int ii = 0; ii < 2; ii++)
			for (int jj = 0; jj < 2; jj++)
				for (int kk = 0; kk < 2; kk++)
					ids->pyz[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;

		/////////////////////////////
		// add pressure pzz
		for (int ii = 0; ii < 2; ii++)
			for (int jj = 0; jj < 2; jj++)
				for (int kk = 0; kk < 2; kk++)
					temp[ii][jj][kk] = part->w[i] * part->w[i] * weight[ii][jj][kk];
		for (int ii = 0; ii < 2; ii++)
			for (int jj = 0; jj < 2; jj++)
				for (int kk = 0; kk < 2; kk++)
					ids->pzz[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
	}
}

/** particle mover  + interpolation*/
int mover_interp(struct particles *part, struct EMfield *field,
								 struct interpDensSpecies *ids, struct grid *grd,
								 struct parameters *param) {
	// print species and subcycling
	printf("***  MOVER + INTERP (FUSED) ITERATIONS = %d - Species %d ***\n", 
			part->NiterMover, part->species_ID);

	// auxiliary variables
	FPpart dt_sub_cycling = (FPpart)param->dt / ((double)part->n_sub_cycles);
	FPpart dto2 = .5 * dt_sub_cycling, qomdt2 = part->qom * dto2 / param->c;
	FPpart omdtsq, denom, ut, vt, wt, udotb;

	// local (to the particle) electric and magnetic field
	FPfield Exl = 0.0, Eyl = 0.0, Ezl = 0.0, Bxl = 0.0, Byl = 0.0, Bzl = 0.0;

	// interpolation densities
	int ix, iy, iz;
	FPfield weight[2][2][2];
	FPfield xi[2], eta[2], zeta[2];

	FPpart temp[2][2][2];

	// intermediate particle position and velocity
	FPpart xptilde, yptilde, zptilde, uptilde, vptilde, wptilde;

	// move each particle with new fields
	for (int i = 0; i < part->nop; i++) {
		xptilde = part->x[i];
		yptilde = part->y[i];
		zptilde = part->z[i];
		uptilde = 0.0;
		vptilde = 0.0;
		wptilde = 0.0;
		// calculate the average velocity iteratively
		for (int innter = 0; innter < part->NiterMover; innter++) {
			// interpolation G-->P
			ix = 2 + (int)((part->x[i] - grd->xStart) * grd->invdx);
			iy = 2 + (int)((part->y[i] - grd->yStart) * grd->invdy);
			iz = 2 + (int)((part->z[i] - grd->zStart) * grd->invdz);

			// calculate weights
			xi[0] = part->x[i] - grd->XN[ix - 1][iy][iz];
			eta[0] = part->y[i] - grd->YN[ix][iy - 1][iz];
			zeta[0] = part->z[i] - grd->ZN[ix][iy][iz - 1];
			xi[1] = grd->XN[ix][iy][iz] - part->x[i];
			eta[1] = grd->YN[ix][iy][iz] - part->y[i];
			zeta[1] = grd->ZN[ix][iy][iz] - part->z[i];
			for (int ii = 0; ii < 2; ii++)
				for (int jj = 0; jj < 2; jj++)
					for (int kk = 0; kk < 2; kk++)
						weight[ii][jj][kk] = xi[ii] * eta[jj] * zeta[kk] * grd->invVOL;

			// set to zero local electric and magnetic field
			Exl = 0.0, Eyl = 0.0, Ezl = 0.0, Bxl = 0.0, Byl = 0.0, Bzl = 0.0;

			for (int ii = 0; ii < 2; ii++)
				for (int jj = 0; jj < 2; jj++)
					for (int kk = 0; kk < 2; kk++) {
						Exl += weight[ii][jj][kk] * field->Ex[ix - ii][iy - jj][iz - kk];
						Eyl += weight[ii][jj][kk] * field->Ey[ix - ii][iy - jj][iz - kk];
						Ezl += weight[ii][jj][kk] * field->Ez[ix - ii][iy - jj][iz - kk];
						Bxl += weight[ii][jj][kk] * field->Bxn[ix - ii][iy - jj][iz - kk];
						Byl += weight[ii][jj][kk] * field->Byn[ix - ii][iy - jj][iz - kk];
						Bzl += weight[ii][jj][kk] * field->Bzn[ix - ii][iy - jj][iz - kk];
					}

			// end interpolation
			omdtsq = qomdt2 * qomdt2 * (Bxl * Bxl + Byl * Byl + Bzl * Bzl);
			denom = 1.0 / (1.0 + omdtsq);
			// solve the position equation
			ut = part->u[i] + qomdt2 * Exl;
			vt = part->v[i] + qomdt2 * Eyl;
			wt = part->w[i] + qomdt2 * Ezl;
			udotb = ut * Bxl + vt * Byl + wt * Bzl;
			// solve the velocity equation
			uptilde =
					(ut + qomdt2 * (vt * Bzl - wt * Byl + qomdt2 * udotb * Bxl)) * denom;
			vptilde =
					(vt + qomdt2 * (wt * Bxl - ut * Bzl + qomdt2 * udotb * Byl)) * denom;
			wptilde =
					(wt + qomdt2 * (ut * Byl - vt * Bxl + qomdt2 * udotb * Bzl)) * denom;
			// update position
			part->x[i] = xptilde + uptilde * dto2;
			part->y[i] = yptilde + vptilde * dto2;
			part->z[i] = zptilde + wptilde * dto2;

		}  // end of iteration
		// update the final position and velocity
		part->u[i] = 2.0 * uptilde - part->u[i];
		part->v[i] = 2.0 * vptilde - part->v[i];
		part->w[i] = 2.0 * wptilde - part->w[i];
		part->x[i] = xptilde + uptilde * dt_sub_cycling;
		part->y[i] = yptilde + vptilde * dt_sub_cycling;
		part->z[i] = zptilde + wptilde * dt_sub_cycling;

		//////////
		//////////
		////////// BC

		// X-DIRECTION: BC particles
		if (part->x[i] > grd->Lx) {
			if (param->PERIODICX == true) {  // PERIODIC
				part->x[i] = part->x[i] - grd->Lx;
			} else {  // REFLECTING BC
				part->u[i] = -part->u[i];
				part->x[i] = 2 * grd->Lx - part->x[i];
			}
		}

		if (part->x[i] < 0) {
			if (param->PERIODICX == true) {  // PERIODIC
				part->x[i] = part->x[i] + grd->Lx;
			} else {  // REFLECTING BC
				part->u[i] = -part->u[i];
				part->x[i] = -part->x[i];
			}
		}

		// Y-DIRECTION: BC particles
		if (part->y[i] > grd->Ly) {
			if (param->PERIODICY == true) {  // PERIODIC
				part->y[i] = part->y[i] - grd->Ly;
			} else {  // REFLECTING BC
				part->v[i] = -part->v[i];
				part->y[i] = 2 * grd->Ly - part->y[i];
			}
		}

		if (part->y[i] < 0) {
			if (param->PERIODICY == true) {  // PERIODIC
				part->y[i] = part->y[i] + grd->Ly;
			} else {  // REFLECTING BC
				part->v[i] = -part->v[i];
				part->y[i] = -part->y[i];
			}
		}

		// Z-DIRECTION: BC particles
		if (part->z[i] > grd->Lz) {
			if (param->PERIODICZ == true) {  // PERIODIC
				part->z[i] = part->z[i] - grd->Lz;
			} else {  // REFLECTING BC
				part->w[i] = -part->w[i];
				part->z[i] = 2 * grd->Lz - part->z[i];
			}
		}

		if (part->z[i] < 0) {
			if (param->PERIODICZ == true) {  // PERIODIC
				part->z[i] = part->z[i] + grd->Lz;
			} else {  // REFLECTING BC
				part->w[i] = -part->w[i];
				part->z[i] = -part->z[i];
			}
		}

		// Here we can do interpolation
		// determine cell: can we change to int()? is it faster?
		ix = 2 + (int)(floor((part->x[i] - grd->xStart) * grd->invdx));
		iy = 2 + (int)(floor((part->y[i] - grd->yStart) * grd->invdy));
		iz = 2 + (int)(floor((part->z[i] - grd->zStart) * grd->invdz));

		// distances from node
		xi[0] = part->x[i] - grd->XN[ix - 1][iy][iz];
		eta[0] = part->y[i] - grd->YN[ix][iy - 1][iz];
		zeta[0] = part->z[i] - grd->ZN[ix][iy][iz - 1];
		xi[1] = grd->XN[ix][iy][iz] - part->x[i];
		eta[1] = grd->YN[ix][iy][iz] - part->y[i];
		zeta[1] = grd->ZN[ix][iy][iz] - part->z[i];

		// calculate the weights for different nodes
		for (int ii = 0; ii < 2; ii++)
			for (int jj = 0; jj < 2; jj++)
				for (int kk = 0; kk < 2; kk++)
					weight[ii][jj][kk] =
							part->q[i] * xi[ii] * eta[jj] * zeta[kk] * grd->invVOL;

		//////////////////////////
		// add charge density
		for (int ii = 0; ii < 2; ii++)
			for (int jj = 0; jj < 2; jj++)
				for (int kk = 0; kk < 2; kk++)
					ids->rhon[ix - ii][iy - jj][iz - kk] +=
							weight[ii][jj][kk] * grd->invVOL;

		////////////////////////////
		// add current density - Jx
		for (int ii = 0; ii < 2; ii++)
			for (int jj = 0; jj < 2; jj++)
				for (int kk = 0; kk < 2; kk++)
					temp[ii][jj][kk] = part->u[i] * weight[ii][jj][kk];

		for (int ii = 0; ii < 2; ii++)
			for (int jj = 0; jj < 2; jj++)
				for (int kk = 0; kk < 2; kk++)
					ids->Jx[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;

		////////////////////////////
		// add current density - Jy
		for (int ii = 0; ii < 2; ii++)
			for (int jj = 0; jj < 2; jj++)
				for (int kk = 0; kk < 2; kk++)
					temp[ii][jj][kk] = part->v[i] * weight[ii][jj][kk];
		for (int ii = 0; ii < 2; ii++)
			for (int jj = 0; jj < 2; jj++)
				for (int kk = 0; kk < 2; kk++)
					ids->Jy[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;

		////////////////////////////
		// add current density - Jz
		for (int ii = 0; ii < 2; ii++)
			for (int jj = 0; jj < 2; jj++)
				for (int kk = 0; kk < 2; kk++)
					temp[ii][jj][kk] = part->w[i] * weight[ii][jj][kk];
		for (int ii = 0; ii < 2; ii++)
			for (int jj = 0; jj < 2; jj++)
				for (int kk = 0; kk < 2; kk++)
					ids->Jz[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;

		////////////////////////////
		// add pressure pxx
		for (int ii = 0; ii < 2; ii++)
			for (int jj = 0; jj < 2; jj++)
				for (int kk = 0; kk < 2; kk++)
					temp[ii][jj][kk] = part->u[i] * part->u[i] * weight[ii][jj][kk];
		for (int ii = 0; ii < 2; ii++)
			for (int jj = 0; jj < 2; jj++)
				for (int kk = 0; kk < 2; kk++)
					ids->pxx[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;

		////////////////////////////
		// add pressure pxy
		for (int ii = 0; ii < 2; ii++)
			for (int jj = 0; jj < 2; jj++)
				for (int kk = 0; kk < 2; kk++)
					temp[ii][jj][kk] = part->u[i] * part->v[i] * weight[ii][jj][kk];
		for (int ii = 0; ii < 2; ii++)
			for (int jj = 0; jj < 2; jj++)
				for (int kk = 0; kk < 2; kk++)
					ids->pxy[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;

		/////////////////////////////
		// add pressure pxz
		for (int ii = 0; ii < 2; ii++)
			for (int jj = 0; jj < 2; jj++)
				for (int kk = 0; kk < 2; kk++)
					temp[ii][jj][kk] = part->u[i] * part->w[i] * weight[ii][jj][kk];
		for (int ii = 0; ii < 2; ii++)
			for (int jj = 0; jj < 2; jj++)
				for (int kk = 0; kk < 2; kk++)
					ids->pxz[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;

		/////////////////////////////
		// add pressure pyy
		for (int ii = 0; ii < 2; ii++)
			for (int jj = 0; jj < 2; jj++)
				for (int kk = 0; kk < 2; kk++)
					temp[ii][jj][kk] = part->v[i] * part->v[i] * weight[ii][jj][kk];
		for (int ii = 0; ii < 2; ii++)
			for (int jj = 0; jj < 2; jj++)
				for (int kk = 0; kk < 2; kk++)
					ids->pyy[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;

		/////////////////////////////
		// add pressure pyz
		for (int ii = 0; ii < 2; ii++)
			for (int jj = 0; jj < 2; jj++)
				for (int kk = 0; kk < 2; kk++)
					temp[ii][jj][kk] = part->v[i] * part->w[i] * weight[ii][jj][kk];
		for (int ii = 0; ii < 2; ii++)
			for (int jj = 0; jj < 2; jj++)
				for (int kk = 0; kk < 2; kk++)
					ids->pyz[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;

		/////////////////////////////
		// add pressure pzz
		for (int ii = 0; ii < 2; ii++)
			for (int jj = 0; jj < 2; jj++)
				for (int kk = 0; kk < 2; kk++)
					temp[ii][jj][kk] = part->w[i] * part->w[i] * weight[ii][jj][kk];
		for (int ii = 0; ii < 2; ii++)
			for (int jj = 0; jj < 2; jj++)
				for (int kk = 0; kk < 2; kk++)
					ids->pzz[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;

		// end of interpolation

	}  // end of particles

	return (0);  // exit succcesfully
}  // end of the mover + interp
