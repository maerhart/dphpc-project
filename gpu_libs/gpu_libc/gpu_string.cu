#include "string.h.cuh"
#include "ctypes.h.cuh"
#include "assert.h.cuh"

__device__ char *strcpy(char *dest, const char *src) {
    for (size_t i = 0; src[i] != '\0'; i++) {
        dest[i] = src[i];
    }
    return dest;
}

__device__ char *strncpy(char *dest, const char *src, size_t n) {
    size_t i = 0;
    while (i < n && src[i] != '\0') {
        dest[i] = src[i];
        i++;
    }
    while (i < n) {
        dest[i] = '\0';
        i++;
    }
    return dest;
}

__device__ char *strcat(char *s, const char *t) {
    char *dest=s;
    s+=strlen(s);
    for (;;) {
        if (!(*s = *t)) break;
        ++s; ++t;
    }
    return dest;
}

__device__ char *strncat(char *s, const char *t, size_t n) {
    char *dest=s;
    register char *max;
    s += strlen(s);
    if (((max=s+n)==s)) goto fini;
    for (;;) {
        if ((!(*s = *t))) break;
        if ((++s==max)) break;
        ++t;
  }
  *s=0;
fini:
    return dest;
}

__device__ size_t strxfrm(char *dest, const char *src, size_t n) {
    memset(dest,0,n);
    memccpy(dest,src,0,n);
    return strlen(dest);
}

__device__ size_t strlen(const char *s) {
    size_t i;
    if (!s) return 0;
    for (i=0; (*s); ++s) ++i;
    return i;
}


__device__ int strcmp(const char *s1, const char *s2) {
    while (*s1 != '\0' && *s1 == *s2) {
        s1++;
        s2++;
    }
    return s1 - s2;
}

__device__ int strncmp(const char *s1, const char *s2, size_t n) {
  const unsigned char* a=(const unsigned char*)s1;
  const unsigned char* b=(const unsigned char*)s2;
  const unsigned char* fini=a+n;
  while (a!=fini) {
    int res=*a-*b;
    if (res) return res;
    if (!*a) return 0;
    ++a; ++b;
  }
  return 0;
}

__device__ int strcoll(const char *s1, const char *s2) {
    return strcmp(s1, s2);
}

__device__ char *strchr(const char *t, int c) {
  char ch;
  ch = c;
  for (;;) {
    if ((*t == ch)) break;
    if ((!*t)) return 0; ++t;
  }
  return (char*)t;
}

__device__ char *strrchr(const char *t, int c) {
  char ch;
  const char *l=0;
  ch = c;
  for (;;) {
    if ((*t == ch)) l=t;
    if ((!*t)) return (char*)l;
    ++t;
  }
  return (char*)l;
}

__device__ size_t strspn(const char *s, const char *accept) {
  size_t l = 0;
  const char *a;
  for (; *s; s++) {
    for (a = accept; *a && *s != *a; a++);
    if (!*a)
      break;
    else
     l++;
  }

  return l;
}

__device__ size_t strcspn(const char *s, const char *reject) {
  size_t l=0;
  int i;
  for (; *s; ++s) {
    for (i=0; reject[i]; ++i)
      if (*s==reject[i]) return l;
    ++l;
  }
  return l;
}

__device__ char *strpbrk(const char *s, const char *accept) {
  unsigned int i;
  for (; *s; s++)
    for (i=0; accept[i]; i++)
      if (*s == accept[i])
	return (char*)s;
  return 0;
}

__device__ char *strstr(const char *haystack, const char *needle) {
  size_t nl=strlen(needle);
  size_t hl=strlen(haystack);
  size_t i;
  if (!nl) goto found;
  if (nl>hl) return 0;
  for (i=hl-nl+1; i; --i) {
    if (*haystack==*needle && !memcmp(haystack,needle,nl))
found:
      return (char*)haystack;
    ++haystack;
  }
  return 0;
}

__device__ char *strtok(char *s, const char *delim) {
  static char *strtok_pos;
  return strtok_r(s,delim,&strtok_pos);
}

__device__ char *strtok_r(char *s, const char *delim, char** ptrptr) {
  char*tmp=0;
  if (s==0) s=*ptrptr;
  s+=strspn(s,delim);
  if ((*s)) {
    tmp=s;
    s+=strcspn(s,delim);
    if ((*s)) *s++=0;
  }
  *ptrptr=s;
  return tmp;
}

__device__ char* strerror(int errnum) {
  //FIXME this is wrong, implement correctly when we have errno, for now this throws warning so it is easy to see
  char message[] = "ERROR!";
  return (char*)message;
}

__device__ void* memset(void * dst, int s, size_t count) {
    char* a = (char*) dst;
    count++;
    while (--count)
	*a++ = s;
    return dst;
}

__device__ void* memcpy (void *dst, const void *src, size_t n) {
    void           *res = dst;
    unsigned char  *c1, *c2;
    c1 = (unsigned char *) dst;
    c2 = (unsigned char *) src;
    while (n--) *c1++ = *c2++;
    return (res);
}

__device__ void *memccpy(void *dst, const void *src, int c, size_t count)
{
  char *a = (char*) dst;
  const char *b = (char*) src;
  while (count--)
  {
    *a++ = *b;
    if (*b==c)
    {
      return (void *)a;
    }
    b++;
  }
  return 0;
}

__device__ void* memmove(void *dst, const void *src, size_t count) {
  char *a = (char*) dst;
  const char *b = (char*) src;
  if (src!=dst)
  {
    if (src>dst)
    {
      while (count--) *a++ = *b++;
    }
    else
    {
      a+=count-1;
      b+=count-1;
      while (count--) *a-- = *b--;
    }
  }
  return dst;
}

__device__ int memcmp(const void *dst, const void *src, size_t count) {
  int r;
  char *d = (char*) dst;
  char *s = (char*) src;
  ++count;
  while ((--count)) {
    if ((r=(*d - *s)))
      return r;
    ++d;
    ++s;
  }
  return 0;
}

__device__ void* memchr(const void *s, int c, size_t n) {
  const unsigned char *pc = (unsigned char *) s;
  for (;n--;pc++)
    if (*pc == c)
      return ((void *) pc);
  return 0;
}

__device__ double atof(const char *nptr) {
  double tmp=strtod(nptr,0);
  return tmp;
}

__device__ int atoi(const char* s) {
  long int v=0;
  int sign=1;
  while ( *s == ' '  ||  (unsigned int)(*s - 9) < 5u) s++;
  switch (*s) {
  case '-': sign=-1;
  case '+': ++s;
  }
  while ((unsigned int) (*s - '0') < 10u) {
    v=v*10+*s-'0'; ++s;
  }
  return sign==-1?-v:v;
}

__device__ long int atol(const char* s) {
  long int v=0;
  int sign=0;
  while ( *s == ' ' || (unsigned int)(*s - 9) < 5u) ++s;
  switch (*s) {
  case '-': sign=-1;
  case '+': ++s;
  }
  while ((unsigned int) (*s - '0') < 10u) {
    v=v*10+*s-'0'; ++s;
  }
  return sign?-v:v;
}

__device__ long long int atoll(const char* s) {
  long long int v=0;
  int sign=1;
  while ( *s == ' '  ||  (unsigned int)(*s - 9) < 5u) ++s;
  switch (*s) {
  case '-': sign=-1;
  case '+': ++s;
  }
  while ((unsigned int) (*s - '0') < 10u) {
    v=v*10+*s-'0'; ++s;
  }
  return sign==-1?-v:v;
}


__device__ float strtof(const char*s , char** endptr) {
    float res;
    res = strtod(s, endptr);
    return res;
}


__device__ double strtod(const char* s, char** endptr) {
    const char*  p     = s;
    float        value = 0.;
    int          sign  = +1;
    float        factor;
    unsigned int expo;

    while ( isspace(*p) )
        p++;

    switch (*p) {
    case '-': sign = -1;
    case '+': p++;
    default : break;
    }

    while ( (unsigned int)(*p - '0') < 10u )
        value = value*10 + (*p++ - '0');

    if ( *p == '.' ) {
        factor = 1.;

        p++;
        while ( (unsigned int)(*p - '0') < 10u ) {
            factor *= 0.1;
            value  += (*p++ - '0') * factor;
        }
    }

    if ( (*p | 32) == 'e' ) {
        expo   = 0;
        factor = 10.L;

        switch (*++p) {
        case '-': factor = 0.1;
        case '+': p++;
                  break;
        case '0': case '1': case '2': case '3': case '4': case '5': case '6': case '7': case '8': case '9':
                  break;
        default : value = 0.L;
                  p     = s;
                  goto done;
        }

        while ( (unsigned int)(*p - '0') < 10u )
            expo = 10 * expo + (*p++ - '0');

        while ( 1 ) {
            if ( expo & 1 )
                value *= factor;
            if ( (expo >>= 1) == 0 )
                break;
            factor *= factor;
        }
    }

done:
    if ( endptr != NULL )
        *endptr = (char*)p;

    return value * sign;
}



__device__ long double strtold(const char* s, char** endptr) {
    return strtod(s, endptr);
}



__device__ long long int strtoll(const char *nptr, char **endptr, int base)
{
  int neg=0;
  unsigned long long int v;
  const char*orig=nptr;

  while(isspace(*nptr)) nptr++;

  if (*nptr == '-' && isalnum(nptr[1])) { neg=-1; nptr++; }
  v=strtoull(nptr,endptr,base);
  if (endptr && *endptr==nptr) *endptr=(char *)orig;
  if (v>LLONG_MAX) {
    if (v==0x8000000000000000ull && neg) {
      errno=0;
      return v;
    }
    return (neg?LLONG_MIN:LLONG_MAX);
  }
  return (neg?-v:v);
}

__device__ long int strtol(const char *nptr, char** endptr, int base) {
    return strtoll(nptr, endptr, base);
}


__device__ unsigned long long int strtoull(const char *ptr, char **endptr, int base)
{
  int neg = 0, overflow = 0;
  long long int v=0;
  const char* orig;
  const char* nptr=ptr;

  while(isspace(*nptr)) ++nptr;

  if (*nptr == '-') { neg=1; nptr++; }
  else if (*nptr == '+') ++nptr;
  orig=nptr;
  if (base==16 && nptr[0]=='0') goto skip0x;
  if (base) {
    register unsigned int b=base-2;
    if ((b>34)) { errno=EINVAL; return 0; }
  } else {
    if (*nptr=='0') {
      base=8;
skip0x:
      if (((*(nptr+1)=='x')||(*(nptr+1)=='X')) && isxdigit(nptr[2])) {
	nptr+=2;
	base=16;
      }
    } else
      base=10;
  }
  while((*nptr)) {
    register unsigned char c=*nptr;
    c=(c>='a'?c-'a'+10:c>='A'?c-'A'+10:c<='9'?c-'0':0xff);
    if ((c>=base)) break;	
    {
      register unsigned long x=(v&0xff)*base+c;
      register unsigned long long w=(v>>8)*base+(x>>8);
      if (w>(ULLONG_MAX>>8)) overflow=1;
      v=(w<<8)+(x&0xff);
    }
    ++nptr;
  }
  if ((nptr==orig)) {
    nptr=ptr;
    errno=EINVAL;
    v=0;
  }
  if (endptr) *endptr=(char *)nptr;
  if (overflow) {
    errno=ERANGE;
    return ULLONG_MAX;
  }
  return (neg?-v:v);
}

__device__ unsigned long long int strtoul(const char *ptr, char **endptr, int base) {
  return strtoul(ptr, endptr, base);
}

