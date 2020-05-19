#ifndef DATATYPES_CUH
#define DATATYPES_CUH

#include <stdbool.h>
#include <stdint.h>

#include "mpi_types_list.cuh"

#define MPI_TYPES_ENUM_F(name, type) name
#define MPI_TYPES_ENUM_SEP ,

enum MPI_Datatype {
    MPI_TYPES_LIST(MPI_TYPES_ENUM_F, MPI_TYPES_ENUM_SEP)
};

#undef MPI_TYPES_ENUM_F
#undef MPI_TYPES_ENUM_SEP

namespace gpu_mpi {


namespace detail {
    
template <MPI_Datatype> struct GetPlainType {};

#define MPI_TYPES_CONV_F(NAME, TYPE) template <> struct GetPlainType<NAME> { using type = TYPE; };
#define MPI_TYPES_CONV_SEP

MPI_TYPES_LIST(MPI_TYPES_CONV_F, MPI_TYPES_CONV_SEP)

#undef MPI_TYPES_CONV_F
#undef MPI_TYPES_CONV_SEP

} // namespace

template <MPI_Datatype T>
using PlainType = typename detail::GetPlainType<T>::type;

__device__ int plainTypeSize(MPI_Datatype type);

}// namespace

#undef MPI_TYPES_LIST

#endif
