#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "cusparse.h"
#include <stdio.h>

namespace Wrapper {
    
    
    double* solveviacuda(
            const double *values, 
            const long long unsigned int  a_nnz, 
            const long long unsigned int  rows, 
            const long long unsigned int  cols, 
            const long long unsigned int * row_ind, 
            const long long unsigned int * col_ptrs,
            double * bpass,
            double * xreturn);
}
