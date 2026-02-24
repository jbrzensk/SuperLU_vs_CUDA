## Makefile: CS_COMP.EXE
## Purpose: Build the `cs_comp.exe` executable which uses CUDA and Armadillo.
##
## Configuration variables below allow overriding CUDA and Armadillo install
## locations without editing recipes. Set these in the environment or at
## invocation time, e.g. `make CUDA_HOME=/usr/local/cuda ARMA_HOME=/usr/local`.

CUDA_HOME ?= /usr
CUDA_INCLUDE := $(CUDA_HOME)/include
CUDA_LIB := $(CUDA_HOME)/lib/x86_64-linux-gnu
CUDA_ARCH ?= sm_89

ARMA_HOME ?= /home/jbrzensk/USERS
ARMA_INCLUDE := $(ARMA_HOME)/include
ARMA_LIB := $(ARMA_HOME)/lib

all: cs_comp.exe

cs_comp.exe: gpusolve.o creatematrixarm.o
	nvcc -o cs_comp.exe gpusolve.o creatematrixarm.o -L$(CUDA_LIB) -lcudart -lcusparse -lcusolver -llapack -lopenblas
    
gpusolve.o:
	nvcc -c -arch=$(CUDA_ARCH) -I$(CUDA_INCLUDE) gpusolve.cu -lcusolver

creatematrixarm.o:
	g++ creatematrixarm.cpp -c -O2 -I$(ARMA_INCLUDE) -DARMA_DONT_USE_WRAPPER -L$(ARMA_LIB) -lopenblas -llapack


clean:
	rm -f *.o cs_comp.exe

