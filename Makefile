VERSION=CPU

CXX=mpicc
CXXFLAGS=-std=c11 -I./include -g -fopenmp

NVCC=nvcc
ARCH=-gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70
NVCCFLAGS=-DMEMCHECK -DUSE_GPU -lineinfo -I./include $(ARCH) -std=c++11 -O3 -g -Xcompiler "-fopenmp -Wall -Wno-unknown-pragmas" --compiler-bindir=$(CXX)

# Default to use host compiler and flags
COMPILER=$(CXX)
FLAGS=$(CXXFLAGS)
TARGET=sputniPIC.out
SRCDIR=src

# Check go GPU or CPU path
ifeq ($(VERSION), GPU)
    FILES=$(shell find $(SRCDIR) -name '*.cu' -o -name '*.c')
    SRCS=$(subst $(SRCDIR)/sputniPIC_CPU.c,,${FILES})

    COMPILER=$(NVCC)
    COMPILER_FLAGS=$(NVCCFLAGS)
else
    FILES=$(shell find $(SRCDIR) -path $(SRCDIR)/gpu -prune -o -name '*.c' -print)
    SRCS=$(subst $(SRCDIR)/sputniPIC_GPU.c,,${FILES})

    COMPILER=$(CXX)
    COMPILER_FLAGS=$(CXXFLAGS)
endif

# Generate list of objects
OBJDIR=src
OBJS=$(subst $(SRCDIR),$(OBJDIR), $(SRCS))
OBJS:=$(subst .c,.o,$(OBJS))
OBJS:=$(subst .cu,.o,$(OBJS))

BIN := ./bin

all: dir $(BIN)/$(TARGET)

dir: ${BIN}

# Create bin folder
${BIN}:
	mkdir -p $(BIN)

# Binary linkage
$(BIN)/$(TARGET): $(OBJS)
	$(COMPILER) $(COMPILER_FLAGS) $+ -o $@ -lm -ldl

# GPU objects
$(SRCDIR)/%.o: $(SRCDIR)/%.cu
	$(NVCC) $(NVCCFLAGS) $< -c -o $@

# Host objects
$(OBJDIR)/%.o: $(SRCDIR)/%.c
	$(COMPILER) $(COMPILER_FLAGS) $< -c -o $@

clean:
	rm -rf $(OBJS)
	rm -rf $(BIN)/*.out
