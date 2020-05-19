// include guards deliberately omitted

#define MPI_TYPES_LIST(F, SEP) \
    F(MPI_CHAR, char) SEP\
    F(MPI_SHORT, signed short int) SEP\
    F(MPI_INT, signed int) SEP\
    F(MPI_LONG, signed long int) SEP\
    F(MPI_LONG_LONG_INT, signed long long int) SEP\
    F(MPI_LONG_LONG, signed long long int) SEP\
    F(MPI_SIGNED_CHAR, signed char) SEP\
    F(MPI_UNSIGNED_CHAR, unsigned char) SEP\
    F(MPI_UNSIGNED_SHORT, unsigned short int) SEP\
    F(MPI_UNSIGNED, unsigned int) SEP\
    F(MPI_UNSIGNED_LONG, unsigned long int) SEP\
    F(MPI_UNSIGNED_LONG_LONG, unsigned long long int) SEP\
    F(MPI_FLOAT, float) SEP\
    F(MPI_DOUBLE, double) SEP\
    F(MPI_LONG_DOUBLE, double) SEP /* there is no long double on GPU */\
    F(MPI_WCHAR, wchar_t) SEP\
    F(MPI_C_BOOL, _Bool) SEP\
    F(MPI_INT8_T, int8_t) SEP\
    F(MPI_INT16_T, int16_t) SEP\
    F(MPI_INT32_T, int32_t) SEP\
    F(MPI_INT64_T, int64_t) SEP\
    F(MPI_UINT8_T, uint8_t) SEP\
    F(MPI_UINT16_T, uint16_t) SEP\
    F(MPI_UINT32_T, uint32_t) SEP\
    F(MPI_UINT64_T, uint64_t) SEP\
    F(MPI_C_COMPLEX, float _Complex) SEP\
    F(MPI_C_FLOAT_COMPLEX, float _Complex) SEP\
    F(MPI_C_DOUBLE_COMPLEX, double _Complex) SEP\
    F(MPI_C_LONG_DOUBLE_COMPLEX, double _Complex) SEP /* there is no long double on GPU */\
    F(MPI_BYTE, char)
