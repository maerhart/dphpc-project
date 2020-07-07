/** A mixed-precision implicit Particle-in-Cell simulator for heterogeneous systems **/

// Allocator for N-D array: chain of pointers
#include "Alloc.h"

// Precision: fix precision for different quantities
#include "PrecisionTypes.h"
// Simulation Parameter - structure
#include "Parameters.h"
// Grid structure
#include "Grid.h"
// Interpolated Quantities Structures
#include "InterpDensSpecies.h"
#include "InterpDensNet.h"
#include "InterpDens_aux.h"

// Field structure
#include "EMfield.h" // Just E and Bn
//#include "EMfield_aux.h" // Bc, Phi, Eth, D

// Particles structure
#include "Particles.h"
//#include "Particles_aux.h" // Needed only if dointerpolation on GPU - avoid reduction on GPU

// solvers
#include "MaxwellSolver.h"

// mover
#include "Mover.h"

// Initial Condition
#include "IC.h"
// Boundary Conditions
#include "BC.h"
// Smoothing
#include "Smoothing.h"
// timing
#include "Timing.h"
// Read and output operations
// #include "RW_IO.h"
#include "IO.h"

#include <omp.h>
#include <mpi.h>

#include "mpi_comm.h"



// ====================================================== //
// Local function declarations

double timer(
    double *mean, 
    double *variance, 
    double *cycle,
    double start_time,
    long count);


// ====================================================== //
// Main


int main(int argc, char **argv){


    // ====================================================== //
    // Init MPI 
	int mpi_thread_support;
	int mpi_rank, mpi_size;
	MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &mpi_thread_support);
	MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
	MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    if(!mpi_rank){
        printf("Total number of MPI ranks: %d\n", mpi_size);
        printf("Number of cores on root: %d\n", omp_get_max_threads());
    }
    

    // ====================================================== //
    // Read the inputfile and fill the param structure

    struct parameters param;
    // readInputFile(&param,argc,argv);
    hardcodedDefaultParameters(&param);
    if(!mpi_rank){
        printParameters(&param);
        // saveParameters(&param);
    }

    // ====================================================== //
    // Declare variables and alloc memory

    // Set-up the grid information
    struct grid grd;
    setGrid(&param, &grd);
    
    // Allocate Fields
    struct EMfield field;
    field_allocate(&grd,&field);
    struct EMfield_aux field_aux;
    field_aux_allocate(&grd,&field_aux);
    
    // Allocate Interpolated Quantities
    // per species
    struct interpDensSpecies ids[PARAMETERS_NS_MAX];
    for (int is=0; is < param.ns; is++)
        interp_dens_species_allocate(&grd,&ids[is],is);
    // Net densities
    struct interpDensNet idn;
    interp_dens_net_allocate(&grd,&idn);
    // Hat densities
    struct interpDens_aux id_aux;
    interp_dens_aux_allocate(&grd,&id_aux);
    
    // Allocate Particles
    struct particles part[PARAMETERS_NS_MAX];
    struct particles part_global[PARAMETERS_NS_MAX];
    
    // allocation for global particles
    if(!mpi_rank){
        for (int is=0; is < param.ns; is++){
            particle_allocate(&param, &part_global[is], is);
        }
    }

    // ====================================================== //
    // Initialization global system

    if(!mpi_rank){
        initGEM(&param,&grd,&field,&field_aux,part_global,ids);
        //initUniform(&params_global,&grd,&field,&field_aux,part_global,ids);
    }

    // ====================================================== //
    // Distribute system to worker processors.
    // We do particle decomposition, shared domain. 

    // Set number of particles per species for local processes
	for (int i = 0; i < param.ns; i++){
        param.npTot[i] = param.np[i];
		//number of particles localy is global number/batch size
		param.np[i] /= (param.number_of_batches*mpi_size);
		// Maximum number of particles is also divided by batchsize,
		// since we do particle decomposition
		param.npMax[i] /= (param.number_of_batches*mpi_size);

        if(!mpi_rank)
            printf("Local number of particles for species %d: %ld (%d batches per node)\n", i, param.np[i], param.number_of_batches);
	}

    // allocation of local particles
    for (int is=0; is < param.ns; is++)
        particle_allocate(&param,&part[is],is);

    mpi_broadcast_field(&grd, &field);

    // ====================================================== //
    // Timing variables
    double iStart = MPI_Wtime();
    double time0 = iStart;
    // avg, variance and last cycle exec time for mover, interpolation, field solver and io, respectively
    double average[4] = {0.,0.,0.,0.};
    double variance[4] = {0.,0.,0.,0.};
    double cycle_time[4] = {0.,0.,0.,0.};
    
    // **********************************************************//
    // **** Start the Simulation!  Cycle index start from 1  *** //
    // **********************************************************//
    for (int cycle = param.first_cycle_n; cycle < (param.first_cycle_n + param.ncycles); cycle++) {

        // ====================================================== //
        // implicit mover

        if(!mpi_rank){
            printf("\n***********************\n");
            printf("   cycle = %d\n", cycle);
            printf("***********************\n");
            printf("*** Particle Mover ***\n");
        }

        time0 = MPI_Wtime();

        // set to zero the densities - needed for interpolation
        setZeroDensities(&idn,ids,&grd,param.ns);

        // #pragma omp parallel for // Requires MPI_THREAD_MULTIPLE support. 
        for (int is=0; is < param.ns; is++){

            int b = batch_update_particles(
                &part_global[is], 
                &part[is],
                &field,
                &ids[is],
                &grd,
                &param,
                param.npTot[is]/(param.number_of_batches*mpi_size),
                param.npTot[is]
                );

            if(!mpi_rank)
            printf("Move and interpolate species %d in %d batches using MPI\n", is, b);

            applyBCids(&ids[is],&grd,&param);
        }

        time0 = timer(&average[0], &variance[0], &cycle_time[0], time0, cycle);

        // sum over species
        sumOverSpecies(&idn,ids,&grd,param.ns);


        // MPI communicate densities to master, both net densities and for species
		mpi_reduce_dens_net(&grd, &idn);
		for(int is=0; is<param.ns; is++){
			mpi_reduce_dens_spec(&grd, &ids[is]);
		}

        // Update timer for interpolation
        time0 = timer(&average[1], &variance[1], &cycle_time[1], time0, cycle);

        // ====================================================== //
        // From here, master calculates new EM field. workers idle

        if(!mpi_rank){
            // interpolate charge density from center to node
            applyBCscalarDensN(idn.rhon,&grd,&param);
            interpN2Cinterp(idn.rhoc,idn.rhon, &grd);

            // ====================================================== //
            // Maxwell solver

            // calculate hat functions rhoh and Jxh, Jyh, Jzh
            calculateHatDensities(&id_aux, &idn, ids, &field, &grd, &param);

            //  Poisson correction
            if (param.PoissonCorrection)
                divergenceCleaning(&grd,&field_aux,&field,&idn,&param);

            calculateE(&grd,&field_aux,&field,&id_aux,ids,&param);
            calculateB(&grd,&field_aux,&field,&param);

        }

        // broadcast EM field data from master process to slaves
        mpi_broadcast_field(&grd, &field);
        // Update timer for field solver
        time0 = timer(&average[2], &variance[2], &cycle_time[2], time0, cycle);
            
        // ====================================================== //
        // IO
        if(!mpi_rank){
            // write E, B, rho to disk
            if (cycle%param.FieldOutputCycle==0){
                VTK_Write_Vectors(cycle, &grd,&field, &param);
                VTK_Write_Scalars(cycle, &grd,ids,&idn, &param);
            }
        }
        // Update timer for io
        time0 = timer(&average[3], &variance[3], &cycle_time[3], time0, cycle);

        if(!mpi_rank)
            printf("Timing Cycle %d : %f, %f, %f, %f\n", cycle, cycle_time[0],cycle_time[1],cycle_time[2],cycle_time[3]);
    }  // end of one PIC cycle
    
    // ====================================================== //
    /// Release the resources

    // deallocate field
    grid_deallocate(&grd);
    field_deallocate(&grd,&field);
    // interp
    interp_dens_net_deallocate(&grd,&idn);
    interp_dens_aux_deallocate(&grd,&id_aux);
    
    // Deallocate interpolated densities and particles
    for (int is=0; is < param.ns; is++){
        interp_dens_species_deallocate(&grd,&ids[is]);
        particle_deallocate(&part[is]);
    }

    // Dealloc global particles array
    if(!mpi_rank){
        for (int is=0; is < param.ns; is++)
            particle_deallocate(&part_global[is]);
    }
    
    if(!mpi_rank){
        // stop timer
        double iElaps = MPI_Wtime() - iStart;

        // Print timing of simulation
        printf("******************************************************\n");
        printf("   Tot. Simulation Time (s) = %f\n", iElaps);
        printf("   Mover Time / Cycle   (s) = %f+-%f\n", average[0], sqrt(variance[0] / (param.ncycles - 1)));
        printf("   Interp. Time / Cycle (s) = %f+-%f\n", average[1], sqrt(variance[1] / (param.ncycles - 1)));
        printf("   Field Time / Cycle   (s) = %f+-%f\n", average[2], sqrt(variance[2] / (param.ncycles - 1)));
        printf("   IO Time / Cycle      (s) = %f+-%f\n", average[3], sqrt(variance[3] / (param.ncycles - 1)));
        printf("******************************************************\n");
    }

    MPI_Finalize();
    // exit
    return 0;
}



void update_statistics(
    double *mean, 
    double *variance, 
    double new_value, 
    long count
    )
{
    double delta = new_value - *mean;
    *mean += delta / (double)count;
    double delta2 = new_value - *mean;
    *variance += delta * delta2;
}


double timer(
    double *mean, 
    double *variance, 
    double *cycle,
    double start_time,
    long count)
{
    double t = MPI_Wtime();
    *cycle = t - start_time;
    update_statistics(mean, variance, *cycle, count);
    return t;
}
