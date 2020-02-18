#include "ctypes.h.cuh"

__device__ int isalnum(int ch) {
  return (unsigned int)((ch | 0x20) - 'a') < 26u  ||
	 (unsigned int)( ch         - '0') < 10u;
}

__device__ int isalpha(int ch) {
  return (unsigned int)((ch | 0x20) - 'a') < 26u;
}

__device__ int islower( int ch ) {
    return (unsigned int) (ch - 'a') < 26u;
}

__device__ int isupper(int c) {
  unsigned char x=c&0xff;
  return (x>='A' && x<='Z') || (x>=192 && x<=222 && x!=215);
}

__device__ int isdigit ( int ch ) {
    return (unsigned int)(ch - '0') < 10u;
}

__device__ int isxdigit( int ch ) {
    return (unsigned int)( ch         - '0') < 10u  ||
           (unsigned int)((ch | 0x20) - 'a') <  6u;
}

__device__ int iscntrl( int ch ) {
    return (unsigned int)ch < 32u  ||  ch == 127;
}

__device__ int isgraph(int x) {
  unsigned char c=x&0xff;
  return (c>=33 && c<=126) || c>=161;
}

__device__ int isspace( int ch ) {
  return (unsigned int)(ch - 9) < 5u  ||  ch == ' ';
}

__device__ int isblank(int ch) {
  return (ch==' ' || ch=='\t');
}

__device__ int isprint(int x) {
  unsigned char c=x&0xff;
  return (c>=32 && c<=126) || (c>=160);
}

__device__ int ispunct( int ch ) {
    return isprint (ch)  &&  !isalnum (ch)  &&  !isspace (ch);
}

__device__ int tolower(int ch) {
  if ( (unsigned int)(ch - 'A') < 26u )
    ch += 'a' - 'A';
  return ch;
}

__device__ int toupper(int ch) {
  if ( (unsigned int)(ch - 'a') < 26u )
    ch += 'A' - 'a';
  return ch;
}

