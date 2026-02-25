## Makefile: CS_COMP.EXE / CS_COMP_DSS.EXE
## Purpose: Build executables that use CUDA and Armadillo for sparse Ax=b solving.
##   *_dss variants use cuDSS (CUDA 12+) for true GPU-side direct solving.
##   Original variants use cusolverSp (hybrid CPU+GPU).
##
## Configuration variables below allow overriding CUDA and Armadillo install
## locations without editing recipes. Set these in the environment or at
## invocation time, e.g. `make CUDA_HOME=/usr/local/cuda ARMA_HOME=/usr/local`.
###############################################################################
# Edit these for your system
CUDA_HOME ?= /usr
CUDA_INCLUDE := $(CUDA_HOME)/include
CUDA_LIB := $(CUDA_HOME)/lib/x86_64-linux-gnu
CUDSS_LIB := $(CUDA_LIB)/libcudss/12
CUDA_ARCH ?= sm_89

ARMA_HOME ?= /home/jbrzensk/USERS
ARMA_INCLUDE := $(ARMA_HOME)/include
ARMA_LIB := $(ARMA_HOME)/lib
SUPERLU_LIB = $(ARMA_LIB)
###############################################################################

# Only build cuDSS targets if the library is present
CUDSS_SO := $(wildcard $(CUDSS_LIB)/libcudss.so)
ifdef CUDSS_SO
  DSS_TARGETS := cs_comp_dss.exe mm_comp_dss.exe
else
  DSS_TARGETS :=
  $(info cuDSS not found at $(CUDSS_LIB) -- skipping DSS targets)
endif

all: cs_comp.exe mm_comp.exe $(DSS_TARGETS)

cs_comp.exe: gpusolve.o creatematrixarm.o
	nvcc -o cs_comp.exe gpusolve.o creatematrixarm.o \
	-L$(CUDA_LIB) -lcudart -lcusparse -lcusolver -llapack -lopenblas \
	-L$(ARMA_LIB) -lsuperlu \
	-Xlinker -rpath -Xlinker $(ARMA_LIB)

mm_comp.exe: gpusolve.o loadmimeticmat.o
	nvcc -o mm_comp.exe gpusolve.o loadmimeticmat.o \
	-L$(CUDA_LIB) -lcudart -lcusparse -lcusolver -llapack -lopenblas \
	-L$(ARMA_LIB) -lsuperlu \
	-Xlinker -rpath -Xlinker $(ARMA_LIB)

cs_comp_dss.exe: gpusolveDSS.o creatematrixarm.o
	nvcc -o cs_comp_dss.exe gpusolveDSS.o creatematrixarm.o \
	-L$(CUDSS_LIB) -L$(CUDA_LIB) -lcudart -lcusparse -lcudss -lcublas -llapack -lopenblas \
	-L$(ARMA_LIB) -lsuperlu \
	-Xlinker -rpath -Xlinker $(CUDSS_LIB) \
	-Xlinker -rpath -Xlinker $(ARMA_LIB)

mm_comp_dss.exe: gpusolveDSS.o loadmimeticmat.o
	nvcc -o mm_comp_dss.exe gpusolveDSS.o loadmimeticmat.o \
	-L$(CUDSS_LIB) -L$(CUDA_LIB) -lcudart -lcusparse -lcudss -lcublas -llapack -lopenblas \
	-L$(ARMA_LIB) -lsuperlu \
	-Xlinker -rpath -Xlinker $(CUDSS_LIB) \
	-Xlinker -rpath -Xlinker $(ARMA_LIB)

# Different gpusolvers based on availablility of cuDSS library
gpusolve.o: gpusolve.cu cudaarmwrappers.h
	nvcc -c -arch=$(CUDA_ARCH) -I$(CUDA_INCLUDE) gpusolve.cu -lcusolver

gpusolveDSS.o: gpusolveDSS.cu cudaarmwrappers.h
	nvcc -c -arch=$(CUDA_ARCH) -I$(CUDA_INCLUDE) gpusolveDSS.cu

creatematrixarm.o: creatematrixarm.cpp cudaarmwrappers.h
	g++ creatematrixarm.cpp -c -O2 \
	-I$(ARMA_INCLUDE) -DARMA_DONT_USE_WRAPPER \
	-L$(ARMA_LIB) -lopenblas -llapack \
	-L$(SUPERLU_LIB) -lsuperlu

loadmimeticmat.o: loadmimeticmat.cpp cudaarmwrappers.h
	g++ loadmimeticmat.cpp -c -O2 \
	-I$(ARMA_INCLUDE) -DARMA_DONT_USE_WRAPPER \
	-L$(ARMA_LIB) -lopenblas -llapack \
	-L$(SUPERLU_LIB) -lsuperlu

clean:
	rm -f *.o cs_comp.exe mm_comp.exe cs_comp_dss.exe mm_comp_dss.exe

