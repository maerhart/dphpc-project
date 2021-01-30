#include "string.cuh"
#include "ctype.cuh"
#include "assert.cuh"

__device__ char *__gpu_strcpy(char *dest, const char *src) {
    for (size_t i = 0; src[i] != '\0'; i++) {
        dest[i] = src[i];
    }
    return dest;
}

__device__ char *__gpu_strncpy(char *dest, const char *src, size_t n) {
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

__device__ char *__gpu_strcat(char *s, const char *t) {
    char *dest=s;
    s+=__gpu_strlen(s);
    for (;;) {
        if (!(*s = *t)) break;
        ++s; ++t;
    }
    return dest;
}

__device__ char *__gpu_strncat(char *s, const char *t, size_t n) {
    char *dest=s;
    register char *max;
    s += __gpu_strlen(s);
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

__device__ size_t __gpu_strxfrm(char *dest, const char *src, size_t n) {
    memset(dest,0,n);
    __gpu_memccpy(dest,src,0,n);
    return __gpu_strlen(dest);
}

__device__ size_t __gpu_strlen(const char *s) {
    size_t i;
    if (!s) return 0;
    for (i=0; (*s); ++s) ++i;
    return i;
}


__device__ int __gpu_strcmp(const char *s1, const char *s2) {
    while (*s1 != '\0' && *s1 == *s2) {
        s1++;
        s2++;
    }
    return s1 - s2;
}

__device__ int __gpu_strncmp(const char *s1, const char *s2, size_t n) {
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

__device__ int __gpu_strcoll(const char *s1, const char *s2) {
    return __gpu_strcmp(s1, s2);
}

__device__ char *__gpu_strchr(const char *t, int c) {
  char ch;
  ch = c;
  for (;;) {
    if ((*t == ch)) break;
    if ((!*t)) return 0; ++t;
  }
  return (char*)t;
}

__device__ char *__gpu_strrchr(const char *t, int c) {
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

__device__ size_t __gpu_strspn(const char *s, const char *accept) {
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

__device__ size_t __gpu_strcspn(const char *s, const char *reject) {
  size_t l=0;
  int i;
  for (; *s; ++s) {
    for (i=0; reject[i]; ++i)
      if (*s==reject[i]) return l;
    ++l;
  }
  return l;
}

__device__ char *__gpu_strpbrk(const char *s, const char *accept) {
  unsigned int i;
  for (; *s; s++)
    for (i=0; accept[i]; i++)
      if (*s == accept[i])
	return (char*)s;
  return 0;
}

__device__ char *__gpu_strstr(const char *haystack, const char *needle) {
  size_t nl=__gpu_strlen(needle);
  size_t hl=__gpu_strlen(haystack);
  size_t i;
  if (!nl) goto found;
  if (nl>hl) return 0;
  for (i=hl-nl+1; i; --i) {
    if (*haystack==*needle && !__gpu_memcmp(haystack,needle,nl))
found:
      return (char*)haystack;
    ++haystack;
  }
  return 0;
}

__device__ char *__gpu_strtok(char *s, const char *delim) {
  static char *strtok_pos;
  return __gpu_strtok_r(s,delim,&strtok_pos);
}

__device__ char *__gpu_strtok_r(char *s, const char *delim, char** ptrptr) {
  char*tmp=0;
  if (s==0) s=*ptrptr;
  s+=__gpu_strspn(s,delim);
  if ((*s)) {
    tmp=s;
    s+=__gpu_strcspn(s,delim);
    if ((*s)) *s++=0;
  }
  *ptrptr=s;
  return tmp;
}

__device__ static const char message[] = "ERROR!";

__device__ char* __gpu_strerror(int errnum) {
  //FIXME this is wrong, implement correctly when we have errno, for now this throws warning so it is easy to see
  return (char*)message;
}

__device__ void* __gpu_memcpy(void *dst, const void *src, size_t n) {
    void           *res = dst;
    unsigned char  *c1, *c2;
    c1 = (unsigned char *) dst;
    c2 = (unsigned char *) src;
    while (n--) *c1++ = *c2++;
    return (res);
}

__device__ void *__gpu_memccpy(void *dst, const void *src, int c, size_t count)
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

__device__ void* __gpu_memmove(void *dst, const void *src, size_t count) {
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

__device__ int __gpu_memcmp(const void *dst, const void *src, size_t count) {
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

__device__ void* __gpu_memchr(const void *s, int c, size_t n) {
  const unsigned char *pc = (unsigned char *) s;
  for (;n--;pc++)
    if (*pc == c)
      return ((void *) pc);
  return 0;
}

__device__ double __gpu_atof(const char *nptr) {
  double tmp=__gpu_strtod(nptr,0);
  return tmp;
}

__device__ int __gpu_atoi(const char* s) {
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

__device__ long int __gpu_atol(const char* s) {
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

__device__ long long int __gpu_atoll(const char* s) {
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


__device__ float __gpu_strtof(const char*s , char** endptr) {
    float res;
    res = __gpu_strtod(s, endptr);
    return res;
}


__device__ double __gpu_strtod(const char* s, char** endptr) {
    const char*  p     = s;
    float        value = 0.;
    int          sign  = +1;
    float        factor;
    unsigned int expo;

    while ( __gpu_isspace(*p) )
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



__device__ long double __gpu_strtold(const char* s, char** endptr) {
    return __gpu_strtod(s, endptr);
}



__device__ long long int __gpu_strtoll(const char *nptr, char **endptr, int base)
{
  int neg=0;
  unsigned long long int v;
  const char*orig=nptr;

  while(__gpu_isspace(*nptr)) nptr++;

  if (*nptr == '-' && __gpu_isalnum(nptr[1])) { neg=-1; nptr++; }
  v=__gpu_strtoull(nptr,endptr,base);
  if (endptr && *endptr==nptr) *endptr=(char *)orig;
  if (v>LLONG_MAX) {
    if (v==0x8000000000000000ull && neg) {
      __gpu_errno=0;
      return v;
    }
    return (neg?LLONG_MIN:LLONG_MAX);
  }
  return (neg?-v:v);
}

__device__ long int __gpu_strtol(const char *nptr, char** endptr, int base) {
    return __gpu_strtoll(nptr, endptr, base);
}


__device__ unsigned long long int __gpu_strtoull(const char *ptr, char **endptr, int base)
{
  int neg = 0, overflow = 0;
  long long int v=0;
  const char* orig;
  const char* nptr=ptr;

  while(__gpu_isspace(*nptr)) ++nptr;

  if (*nptr == '-') { neg=1; nptr++; }
  else if (*nptr == '+') ++nptr;
  orig=nptr;
  if (base==16 && nptr[0]=='0') goto skip0x;
  if (base) {
    register unsigned int b=base-2;
    if ((b>34)) { __gpu_errno=EINVAL; return 0; }
  } else {
    if (*nptr=='0') {
      base=8;
skip0x:
      if (((*(nptr+1)=='x')||(*(nptr+1)=='X')) && __gpu_isxdigit(nptr[2])) {
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
    __gpu_errno=EINVAL;
    v=0;
  }
  if (endptr) *endptr=(char *)nptr;
  if (overflow) {
    __gpu_errno=ERANGE;
    return ULLONG_MAX;
  }
  return (neg?-v:v);
}

__device__ unsigned long int __gpu_strtoul(const char *ptr, char **endptr, int base) {
  return __gpu_strtoull(ptr, endptr, base);
}

__device__ char *__gpu_strdup(const char *s) {
  size_t len = __gpu_strlen(s);
  char* copy = (char*) malloc(len + 1);
  memcpy(copy, s, len + 1);
  return copy;
}