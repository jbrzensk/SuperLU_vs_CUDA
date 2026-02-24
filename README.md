# SuperLU_vs_CUDA
A comparison of solving sparse matrices with  SuperLU versus CUDA.

This will eventually compare solving MOLE operators with CUDA equivalents. The real key here is Armadillo (SuperLU) will let you save a factorization, making future solves very efficient. While a LU matrix for a sparse matrix is not necessarily sparse, it is **sparse enough** to fit into memory for very large matrices (1Mx1M).

The number of elements in each direction can be set with the first lines in ```creatematrixarm.cpp``` line

```c++
int size = 4096;
```
which will create a 4096x4096 sprase matrix witha 3% infill of random numbers.

## Build the program
The included makefile has some variables for locations of CUDA, Armadillo, and the SuperLU library install locations, edit these for your system.

```bash
CUDA_HOME ?= /usr
CUDA_INCLUDE := $(CUDA_HOME)/include
CUDA_LIB := $(CUDA_HOME)/lib/x86_64-linux-gnu
CUDA_ARCH ?= sm_89

ARMA_HOME ?= /home/jbrzensk/USERS
ARMA_INCLUDE := $(ARMA_HOME)/include
ARMA_LIB := $(ARMA_HOME)/lib
SUPERLU_LIB = $(ARMA_LIB)
```

Build the program with ```make ```.

## Run the program
Run the executable **cs_comp.exe** from the command line by

``` bash
./cs_comp.exe
```

The pogram will compare an assortment of different techniques for solving sparse systems.

```bash
**********************************************************
Creating a random sparse matrix of size 4096x4096 with 3% density.
**********************************************************

**********************************************************
Doing QR decomposition...
**********************************************************
Time taken by Armadillo QR solve:3.84632 seconds

**********************************************************
Doing SuperLU solve...
**********************************************************
Armadillo supermatrix nrow: 4096
Armadillo supermatrix ncols: 4096
Time taken by SuperLU:19.9945 seconds

**********************************************************
Doing SuperLU Factorization and Solve...
**********************************************************
Armadillo supermatrix nrow: 4096
Armadillo supermatrix ncols: 4096
Time taken by factorization:21.3219 seconds
Reciprocal of condition number: 1.38269e-05
Time taken by SuperLU factored:0.043694 seconds
Time taken by SuperLU factored secodn time:0.04336 seconds
L1 Norm of both the armadillo computations:6.02615e-09

**********************************************************
Doing cusparse solve...
**********************************************************
cusparse time to solve:  2.1 s 
Total time take from ENTIRE CUDA function call:2.44545 seconds
L1 Norm of Armadillo compared with Cuda computations:6.32133e-09
```