#include "mpi_comm.h"
#include <string.h>


// template <typename T>
// MPI_Datatype _mpi_get_basetype(){
// 	if(typeid(T) == typeid(float)){
// 		return MPI_FLOAT;
// 	}
// 	else if(typeid(T) == typeid(double)){
// 		return MPI_DOUBLE;
// 	}
// 	else{
// 		printf("Unknown precision type '%s'", typeid(T).name());
// 		exit(EXIT_FAILURE);
// 	}
// }


void _scatter(FPpart *send, FPpart *recv, long count, MPI_Datatype type){

	MPI_Scatter(
		send, count, type,
		recv, count, type,
		0, MPI_COMM_WORLD
		);

}

void mpi_scatter_particles(struct particles *part_global, struct particles *part_local){

	MPI_Datatype type;
	if(sizeof(FPpart) == sizeof(float)){
		type = MPI_FLOAT;
	}
	else{
		type = MPI_DOUBLE;
	}
	// MPI_Datatype type = _mpi_get_basetype<FPpart>();

	_scatter(part_global->x, part_local->x, part_local->nop, type);
	_scatter(part_global->y, part_local->y, part_local->nop, type);
	_scatter(part_global->z, part_local->z, part_local->nop, type);
	_scatter(part_global->u, part_local->u, part_local->nop, type);
	_scatter(part_global->v, part_local->v, part_local->nop, type);
	_scatter(part_global->w, part_local->w, part_local->nop, type);

	if(sizeof(FPinterp) == sizeof(float)){
		type = MPI_FLOAT;
	}
	else{
		type = MPI_DOUBLE;
	}
	MPI_Scatter(
		part_global->q, part_local->nop, type,
		part_local->q, part_local->nop, type,
		0, MPI_COMM_WORLD
	);

	MPI_Scatter(
		part_global->track_particle, part_local->nop, MPI_C_BOOL,
		part_local->track_particle, part_local->nop, MPI_C_BOOL,
		0, MPI_COMM_WORLD
	);

}


void _reduce_copy(FPinterp* array, FPinterp* recv_buf, int length){

	// MPI_Datatype type = _mpi_get_basetype<FPinterp>();
	MPI_Datatype type;
	if(sizeof(FPinterp) == sizeof(float)){
		type = MPI_FLOAT;
	}
	else{
		type = MPI_DOUBLE;
	}
	MPI_Reduce(array, recv_buf, length, type, MPI_SUM, 0, MPI_COMM_WORLD);
	
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	if(rank == 0){
		memcpy(array, recv_buf, sizeof(FPinterp)*length);
	}
}

void mpi_reduce_dens_net(struct grid* grd, struct interpDensNet* idn){

	int count = grd->nxn*grd->nyn*grd->nzn;
	FPinterp* recv_buf = (FPinterp*) malloc(sizeof(FPinterp)*count);

	_reduce_copy(idn->rhon_flat, recv_buf, count);
	_reduce_copy(idn->rhoc_flat, recv_buf, grd->nxc*grd->nyc*grd->nzc);

	_reduce_copy(idn->Jx_flat, recv_buf, count);
	_reduce_copy(idn->Jy_flat, recv_buf, count);
	_reduce_copy(idn->Jz_flat, recv_buf, count);

	_reduce_copy(idn->pxx_flat, recv_buf, count);
	_reduce_copy(idn->pxy_flat, recv_buf, count);
	_reduce_copy(idn->pxz_flat, recv_buf, count);

	_reduce_copy(idn->pyy_flat, recv_buf, count);
	_reduce_copy(idn->pyz_flat, recv_buf, count);
	_reduce_copy(idn->pzz_flat, recv_buf, count);

	free(recv_buf);

}


void mpi_reduce_dens_spec(struct grid* grd, struct interpDensSpecies* ids){

	int count = grd->nxn*grd->nyn*grd->nzn;
	FPinterp* recv_buf = (FPinterp*) malloc(sizeof(FPinterp)*count);

	_reduce_copy(ids->rhon_flat, recv_buf, count);
	_reduce_copy(ids->rhoc_flat, recv_buf, grd->nxc*grd->nyc*grd->nzc);

	_reduce_copy(ids->Jx_flat, recv_buf, count);
	_reduce_copy(ids->Jy_flat, recv_buf, count);
	_reduce_copy(ids->Jz_flat, recv_buf, count);

	_reduce_copy(ids->pxx_flat, recv_buf, count);
	_reduce_copy(ids->pxy_flat, recv_buf, count);
	_reduce_copy(ids->pxz_flat, recv_buf, count);

	_reduce_copy(ids->pyy_flat, recv_buf, count);
	_reduce_copy(ids->pyz_flat, recv_buf, count);
	_reduce_copy(ids->pzz_flat, recv_buf, count);

	free(recv_buf);

}


void mpi_broadcast_field(struct grid *grd, struct EMfield *field){

	int count = grd->nxn*grd->nyn*grd->nzn;

	// MPI_Datatype type = _mpi_get_basetype<FPinterp>();
	MPI_Datatype type;
	if(sizeof(FPinterp) == sizeof(float)){
		type = MPI_FLOAT;
	}
	else{
		type = MPI_DOUBLE;
	}

	MPI_Bcast(field->Ex_flat, count, type, 0, MPI_COMM_WORLD);
	MPI_Bcast(field->Ey_flat, count, type, 0, MPI_COMM_WORLD);
	MPI_Bcast(field->Ez_flat, count, type, 0, MPI_COMM_WORLD);

	MPI_Bcast(field->Bxn_flat, count, type, 0, MPI_COMM_WORLD);
	MPI_Bcast(field->Byn_flat, count, type, 0, MPI_COMM_WORLD);
	MPI_Bcast(field->Bzn_flat, count, type, 0, MPI_COMM_WORLD);


}