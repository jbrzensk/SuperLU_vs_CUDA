/* cudaarmwrappers.h
 * This header file declares the function solveviacuda which is defined in 
 * gpusolve.cu. This function takes in a sparse matrix in CSC format, a 
 * right-hand side vector, and returns the solution to the linear system 
 * Ax=b using CUDA's cusolver library. The function is designed to be 
 * called from creatematrixarm.cpp which generates the sparse matrix 
 * and right-hand side vector.
 */
 
#include <cuda_runtime.h>

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
