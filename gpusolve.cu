#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "cusolverSp.h"
#include "cusparse.h"
#include "cudaarmwrappers.h"
#include <sys/time.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t _err = (call); \
        if (_err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(_err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define CUSPARSE_CHECK(call) \
    do { \
        cusparseStatus_t _err = (call); \
        if (_err != CUSPARSE_STATUS_SUCCESS) { \
            fprintf(stderr, "cuSPARSE error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cusparseGetErrorString(_err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define CUSOLVER_CHECK(call) \
    do { \
        cusolverStatus_t _err = (call); \
        if (_err != CUSOLVER_STATUS_SUCCESS) { \
            fprintf(stderr, "cuSOLVER error at %s:%d: %d\n", \
                    __FILE__, __LINE__, (int)_err); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

namespace Wrapper {

    double* solveviacuda(
            const double *values,
            const long long unsigned int  a_nnz,
            const long long unsigned int  rows,
            const long long unsigned int  cols,
            const long long unsigned int * row_ind,
            const long long unsigned int * col_ptrs,
            double * bpass,
            double * xreturn){

        if (a_nnz == 0 || rows == 0 || cols == 0) {
            fprintf(stderr, "solveviacuda: invalid dimensions (nnz=%llu rows=%llu cols=%llu)\n",
                    a_nnz, rows, cols);
            return nullptr;
        }

        //Values for timing
        struct timeval t1, t2;

        //Take input from armadillo and convert it to CUDA required data types
        int       nnz = (int)a_nnz;//number of nonzero elements
        double*   h_cscVal;
        int*      h_csccol_pts;
        int*      h_cscRowInd;

        //Need to transfer data due to cast
        h_csccol_pts = (int *)malloc(sizeof(int) * (cols+1));
        for (int i=0 ; i < (cols+1) ; i++){
            h_csccol_pts[i] = (int)col_ptrs[i];
        }

        h_cscRowInd = (int *)malloc(sizeof(int) * nnz);
        for (int i=0 ; i<nnz ; i++){
            h_cscRowInd[i] = (int)row_ind[i];
        }

        //Since values are of same size a simple memcpy works
        h_cscVal = (double *)malloc(sizeof(double) * nnz);
        memcpy(h_cscVal,values,sizeof(double)*nnz);

        //Required values for CSR2CSC
        size_t h_buffer = 0;    //size of workspace
        double *d_buffer = NULL; //device workspace for CSR2CSC

        double * d_cscVal = NULL;
        int * d_csccol_pts = NULL;
        int * d_cscRowInd = NULL;
        double * d_csrvalues = NULL;
        int * d_csrRowPtr = NULL;
        int * d_csrColInd = NULL;

        cusparseHandle_t  sphandle = NULL;
        CUSPARSE_CHECK(cusparseCreate(&sphandle));

        //Allocate for both CSC and CSR formats
        CUDA_CHECK(cudaMalloc((void**)&d_cscVal, sizeof(double) * nnz));
        CUDA_CHECK(cudaMalloc((void**)&d_csccol_pts, sizeof(int) * (cols+1)));
        CUDA_CHECK(cudaMalloc((void**)&d_cscRowInd, sizeof(int) * nnz));
        CUDA_CHECK(cudaMalloc((void**)&d_csrvalues, sizeof(double) * nnz));
        CUDA_CHECK(cudaMalloc((void**)&d_csrRowPtr, sizeof(int) * (rows+1)));
        CUDA_CHECK(cudaMalloc((void**)&d_csrColInd, sizeof(int) * nnz));

        //For CSR2CSC
        cusparseIndexBase_t  idxBase = CUSPARSE_INDEX_BASE_ZERO;
        cusparseAction_t     copyValues = CUSPARSE_ACTION_NUMERIC; // or numeric
        cusparseCsr2CscAlg_t alg = CUSPARSE_CSR2CSC_ALG1;
        cudaDataType         valType = CUDA_R_64F;

        CUDA_CHECK(cudaMemcpy(d_cscVal, h_cscVal, sizeof(double)*nnz, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_csccol_pts, h_csccol_pts, sizeof(int)*(cols+1), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_cscRowInd, h_cscRowInd, sizeof(int)*nnz, cudaMemcpyHostToDevice));

        cusparseMatDescr_t descrA = NULL;
        CUSPARSE_CHECK(cusparseCreateMatDescr(&descrA));
        CUSPARSE_CHECK(cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL));
        CUSPARSE_CHECK(cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO));

        //Compute buffersize required by CSR2CSC
        CUSPARSE_CHECK(cusparseCsr2cscEx2_bufferSize(
                sphandle,
                cols,//h
                rows,//h
                nnz,//h
                d_cscVal,//csrVal originally on device
                d_csccol_pts,//Csrrowptrs originally on device
                d_cscRowInd,//csrcolind orriginally
                d_csrvalues,
                d_csrRowPtr,
                d_csrColInd,
                valType,//valtype
                copyValues,
                idxBase,
                alg,
                &h_buffer));

        CUDA_CHECK(cudaDeviceSynchronize());
        //I don't know why cuda cant just allocate this on the buffer in the above function.
        CUDA_CHECK(cudaMalloc((void**)&d_buffer, sizeof(double)*h_buffer));

        //Perform actual conversion
        /*The documentation states that this function will also work to convert CSC2CSR
          but provides no explaination of how. It states that CSR is really just the transpose of
          CSC so this function will work for either one. It performs this by just copying over the col
          and row indices and REARRANGING THE ELEMENT VALUES. This is not intuitive. And a lesson learned.
         */
        CUSPARSE_CHECK(cusparseCsr2cscEx2(
                sphandle,
                cols,//Cols and rows are switched here because its csc 2 csr
                rows,
                nnz,
                d_cscVal,//unlike documentation csc comes first because we are doing CSC2CSR
                d_csccol_pts,
                d_cscRowInd,
                d_csrvalues,
                d_csrRowPtr,
                d_csrColInd,
                valType,
                copyValues,
                idxBase,
                alg,
                d_buffer));

        CUDA_CHECK(cudaDeviceSynchronize());

        //cusolver and cusparse are different libraries and need a different handle.
        cusolverSpHandle_t solhandle = NULL;
        CUSOLVER_CHECK(cusolverSpCreate(&solhandle));

        //Allocate for b of Ax=b on the device.
        double * d_b = NULL;
        CUDA_CHECK(cudaMalloc((void**)&d_b, sizeof(double)*rows));

        //Required for QR solver
        double tol = 1.e-12;
        const int reorder = 1; /*  reordering */
        int singularity = 0; /* -1 if A is invertible under tol. */

        //Allocate for x of Ax=b on the device
        double * d_x = NULL;
        CUDA_CHECK(cudaMalloc((void**)&d_x, sizeof(double)*rows));

        //copy values from host to device
        CUDA_CHECK(cudaMemcpy(d_b, bpass, sizeof(double)*cols, cudaMemcpyHostToDevice));

        cudaDeviceSynchronize();//For timing

        gettimeofday(&t1, 0);
        CUSOLVER_CHECK(cusolverSpDcsrlsvqr(
                solhandle,
                cols,
                nnz,
                descrA,
                d_csrvalues,
                d_csrRowPtr,
                d_csrColInd,
                d_b,
                tol,
                reorder,
                d_x,
                &singularity));

        cudaDeviceSynchronize();//for timing
        gettimeofday(&t2, 0);

        if (0 <= singularity)
        {
            printf("WARNING: the matrix is singular at row %d under tol (%E)\n", singularity, tol);
        }

        double time = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1e6;
        printf("cusparse time to solve:  %3.1f s \n", time);

        xreturn   = (double*)malloc(sizeof(double)*cols);
        CUDA_CHECK(cudaMemcpy(xreturn, d_x, sizeof(double)*cols, cudaMemcpyDeviceToHost));

        if (solhandle) { cusolverSpDestroy(solhandle); }
        if (sphandle) { cusparseDestroy(sphandle); }
        if (descrA) { cusparseDestroyMatDescr(descrA); }

        if (h_cscVal  ) { free(h_cscVal); }
        if (h_csccol_pts) { free(h_csccol_pts); }
        if (h_cscRowInd) { free(h_cscRowInd); }

        if (d_cscVal   ) { cudaFree(d_cscVal); }
        if (d_csccol_pts) { cudaFree(d_csccol_pts); }
        if (d_cscRowInd) { cudaFree(d_cscRowInd); }
        if (d_csrvalues   ) { cudaFree(d_csrvalues); }
        if (d_csrRowPtr) { cudaFree(d_csrRowPtr); }
        if (d_csrColInd) { cudaFree(d_csrColInd); }
        if (d_buffer) { cudaFree(d_buffer); }
        if (d_x) { cudaFree(d_x); }
        if (d_b) { cudaFree(d_b); }

        return xreturn;

    }


}
