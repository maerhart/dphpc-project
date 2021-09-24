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



void _sendrecv_particles(FPpart *send, FPpart *recv, int len, int send_rank, int recv_rank, int rank, MPI_Request *request, int tag){

	MPI_Datatype type;
	if(sizeof(FPinterp) == sizeof(float)){
		type = MPI_FLOAT;
	}
	else{
		type = MPI_DOUBLE;
	}

	if(rank == send_rank){
		MPI_Send(send, len, type, recv_rank, tag, MPI_COMM_WORLD);
	}
	else if(rank == recv_rank){
		MPI_Recv(recv, len, type, send_rank, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}

}

void send_particle_batch(
	struct particles *part_send, 
	struct particles *part_recv, 
	size_t send_offset, 
	size_t recv_offset, 
	size_t len, 
	int send_rank,
	int recv_rank, 
	int rank
	){

	MPI_Request *request;

	_sendrecv_particles(part_send->x+send_offset, part_recv->x+recv_offset, len, send_rank, recv_offset, rank, request, 0);
	_sendrecv_particles(part_send->y+send_offset, part_recv->y+recv_offset, len, send_rank, recv_offset, rank, request, 1);
	_sendrecv_particles(part_send->z+send_offset, part_recv->z+recv_offset, len, send_rank, recv_offset, rank, request, 2);
	_sendrecv_particles(part_send->u+send_offset, part_recv->u+recv_offset, len, send_rank, recv_offset, rank, request, 3);
	_sendrecv_particles(part_send->v+send_offset, part_recv->v+recv_offset, len, send_rank, recv_offset, rank, request, 4);
	_sendrecv_particles(part_send->w+send_offset, part_recv->w+recv_offset, len, send_rank, recv_offset, rank, request, 5);
	
	MPI_Datatype type;
	if(sizeof(FPinterp) == sizeof(float)){
		type = MPI_FLOAT;
	}
	else{
		type = MPI_DOUBLE;
	}

	if(rank == send_rank){
		MPI_Send(part_send->q+send_offset, len, type, recv_rank, 6, MPI_COMM_WORLD);
	}
	else if(rank == recv_rank){
		MPI_Recv(part_recv->q+recv_offset, len, type, send_rank, 6, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}


}


void _scatter(FPpart *send, FPpart *recv, int *displ, int *count, int rank, MPI_Datatype type){

	MPI_Scatterv(
		send, count, displ, type,
		recv, count[rank], type,
		0, MPI_COMM_WORLD
		);

}

int mpi_scatter_particles(struct particles *part_global, struct particles *part_local, int offset, int batchsize, int num_particles){

	int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	int* counts = malloc(sizeof(int) * size);
	int* displ = malloc(sizeof(int) * size);
	int batch_sum = 0;

	for(int i=0; i<size; i++){
		if(offset + batchsize >= num_particles){
			batchsize = num_particles - offset;
		}
		displ[i] = offset;
		counts[i] = batchsize;
		offset += batchsize;
		batch_sum += batchsize;
	}

	MPI_Datatype type;
	if(sizeof(FPpart) == sizeof(float)){
		type = MPI_FLOAT;
	}
	else{
		type = MPI_DOUBLE;
	}
	// MPI_Datatype type = _mpi_get_basetype<FPpart>();

	_scatter(part_global->x, part_local->x, displ, counts, rank, type);
	_scatter(part_global->y, part_local->y, displ, counts, rank, type);
	_scatter(part_global->z, part_local->z, displ, counts, rank, type);
	_scatter(part_global->u, part_local->u, displ, counts, rank, type);
	_scatter(part_global->v, part_local->v, displ, counts, rank, type);
	_scatter(part_global->w, part_local->w, displ, counts, rank, type);

	if(sizeof(FPinterp) == sizeof(float)){
		type = MPI_FLOAT;
	}
	else{
		type = MPI_DOUBLE;
	}
	MPI_Scatterv(
		part_global->q, counts, displ, type,
		part_local->q, counts[rank], type,
		0, MPI_COMM_WORLD
		);

	// MPI_Scatterv(
	// 	part_global->track_particle, counts, displ, MPI_C_BOOL,
	// 	part_local->track_particle, counts[rank], MPI_C_BOOL,
	// 	0, MPI_COMM_WORLD
	// 	);

	part_local->nop = counts[rank];

    free(counts);
    free(displ);

	return batch_sum;

}

void _gather(FPpart *send, FPpart *recv, int *displ, int *count, int rank, MPI_Datatype type){

	MPI_Gatherv(
		send, count[rank], type,
		recv, count, displ, type,
		0, MPI_COMM_WORLD
		);

}

void mpi_gather_particles(struct particles *part_global, struct particles *part_local, int offset, int batchsize, int num_particles){

	int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	int* counts = malloc(sizeof(int) * size);
	int* displ = malloc(sizeof(int) * size);

	for(int i=0; i<size; i++){
		if(offset + batchsize >= num_particles){
			batchsize = num_particles - offset;
		}
		displ[i] = offset;
		counts[i] = batchsize;
		offset += batchsize;
	}


	MPI_Datatype type;
	if(sizeof(FPpart) == sizeof(float)){
		type = MPI_FLOAT;
	}
	else{
		type = MPI_DOUBLE;
	}
	// MPI_Datatype type = _mpi_get_basetype<FPpart>();

	_gather(part_local->x, part_global->x, displ, counts, rank, type);
	_gather(part_local->y, part_global->y, displ, counts, rank, type);
	_gather(part_local->z, part_global->z, displ, counts, rank, type);
	_gather(part_local->u, part_global->u, displ, counts, rank, type);
	_gather(part_local->v, part_global->v, displ, counts, rank, type);
	_gather(part_local->w, part_global->w, displ, counts, rank, type);

	if(sizeof(FPinterp) == sizeof(float)){
		type = MPI_FLOAT;
	}
	else{
		type = MPI_DOUBLE;
	}
	MPI_Gatherv(
		part_local->q, counts[rank], type,
		part_global->q, counts, displ, type,
		0, MPI_COMM_WORLD
		);

	// MPI_Scatterv(
	// 	part_local->track_particle, counts, displ, MPI_C_BOOL,
	// 	part_global->track_particle, counts[rank], MPI_C_BOOL,
	// 	0, MPI_COMM_WORLD
	// 	);
    
    free(counts);
    free(displ);

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
