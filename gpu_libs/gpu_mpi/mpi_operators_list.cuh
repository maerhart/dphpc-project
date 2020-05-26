// include guards deliberately omitted


#define MPI_OPERATORS_LIST(F, SEP) \
    F(   MPI_MAX,    OpMax) SEP\
    F(   MPI_MIN,    OpMin) SEP\
    F(   MPI_SUM,    OpSum) SEP\
    F(  MPI_PROD,   OpProd) SEP\
    F(  MPI_LAND,   OpLAnd) SEP\
    F(  MPI_BAND,   OpBAnd) SEP\
    F(   MPI_LOR,    OpLOr) SEP\
    F(   MPI_BOR,    OpBOr) SEP\
    F(  MPI_LXOR,   OpLXor) SEP\
    F(  MPI_BXOR,   OpBXor) SEP\
    F(MPI_MAXLOC, OpMaxLoc) SEP\
    F(MPI_MINLOC, OpMinLoc)











