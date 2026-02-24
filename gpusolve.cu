
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda_runtime.h>
#include "cusolverSp.h"
#include "cusparse.h"
#include "cudaarmwrappers.h"
#include <sys/time.h>





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

        //Print Statement to verify casting worked 
        /* 
           printf("CSC Known Good Input:\n");
           for(int j = 0 ; j < nnz; j++){
           printf("(%llu, %llu, %f) \n", row_ind[j], col_ptrs[row_ind[j]],  values[j] );
           }


           printf("After cast, Before CSC to CSR\n");
           for(int j = 0 ; j < nnz; j++){
           printf("(%d, %d, %f) \n", h_cscRowInd[j], h_csccol_pts[h_cscRowInd[j]], h_cscVal[j] );
           }
         */


        //Required values for CSR2CSC
        size_t h_buffer = 0;    //size of workspace
        double *d_buffer = NULL; //device workspace for CSR2CSC

        double * d_cscVal = NULL;
        int * d_csccol_pts = NULL;
        int * d_cscRowInd = NULL;
        double * d_csrvalues = NULL;
        int * d_csrRowPtr = NULL;
        int * d_csrColInd = NULL;

        cusparseStatus_t status = CUSPARSE_STATUS_SUCCESS; 
        cusparseHandle_t  sphandle = NULL;
        status = cusparseCreate(&sphandle);
        assert(CUSPARSE_STATUS_SUCCESS == status);

        cudaError_t cudaStat1 = cudaSuccess;
        cudaError_t cudaStat2 = cudaSuccess;
        cudaError_t cudaStat3 = cudaSuccess;
        cudaError_t cudaStat4 = cudaSuccess;
        cudaError_t cudaStat5 = cudaSuccess;
        cudaError_t cudaStat6 = cudaSuccess;

        //Allocate for both CSC and CSR formats
        cudaStat1 = cudaMalloc ((void**)&d_cscVal, sizeof(double) * nnz );
        cudaStat3 = cudaMalloc ((void**)&d_csccol_pts, sizeof(int) * (cols+1));
        cudaStat4 = cudaMalloc ((void**)&d_cscRowInd, sizeof( int) * nnz);
        cudaStat2 = cudaMalloc ((void**)&d_csrvalues, sizeof(double) * nnz);
        cudaStat5 = cudaMalloc ((void**)&d_csrRowPtr, sizeof(int) * (rows+1));
        cudaStat6 = cudaMalloc ((void**)&d_csrColInd, sizeof(int) * nnz);
        assert(cudaSuccess == cudaStat1);
        assert(cudaSuccess == cudaStat2);
        assert(cudaSuccess == cudaStat3);
        assert(cudaSuccess == cudaStat4);
        assert(cudaSuccess == cudaStat5);
        assert(cudaSuccess == cudaStat6);

        //For CSR2CSC
        cusparseIndexBase_t  idxBase = CUSPARSE_INDEX_BASE_ZERO;
        cusparseAction_t     copyValues = CUSPARSE_ACTION_NUMERIC; // or numeric 
        cusparseCsr2CscAlg_t alg = CUSPARSE_CSR2CSC_ALG1;   
        cudaDataType         valType = CUDA_R_64F;


        cudaStat1 = cudaMemcpy(d_cscVal, h_cscVal, sizeof(double)*nnz, cudaMemcpyHostToDevice);
        cudaStat1 = cudaMemcpy(d_csccol_pts, h_csccol_pts, sizeof(int)*(cols+1), cudaMemcpyHostToDevice);
        cudaStat1 = cudaMemcpy(d_cscRowInd, h_cscRowInd, sizeof(int)*nnz, cudaMemcpyHostToDevice);

        cusparseMatDescr_t descrA = NULL;
        cusparseCreateMatDescr(&descrA);
        cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
        cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);  

        //Compute buffersize required by CSR2CSC
        status = cusparseCsr2cscEx2_bufferSize( 
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
                &h_buffer);

        assert(CUSPARSE_STATUS_SUCCESS == status);

        cudaStat1 = cudaDeviceSynchronize();     
        assert(cudaSuccess == cudaStat1);
        //I don't know why cuda cant just allocate this on the buffer in the above function.
        cudaStat1 = cudaMalloc((void**)&d_buffer, sizeof(double)*h_buffer);
        assert(cudaSuccess == cudaStat1);

        //Perform actual conversion
        /*The documentation states that this function will also work to convert CSC2CSR
          but provides no explaination of how. It states that CSR is really just the transpose of
          CSC so this function will work for either one. It performs this by just copying over the col
          and row indices and REARRANGING THE ELEMENT VALUES. This is not intuitive. And a lesson learned. 
         */
        status = cusparseCsr2cscEx2(
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
                d_buffer);

        assert(CUSPARSE_STATUS_SUCCESS == status);


        cudaStat1 = cudaDeviceSynchronize();     
        assert(cudaSuccess == cudaStat1);

        //Copy values back to host to see if CSR2CSC worked
        /*
        double  * h_csrvalues;
        int     * h_csrRowPtr;
        int     * h_csrColInd;

        h_csrvalues = (double *)malloc(sizeof(double) * nnz); 
        h_csrRowPtr = (int *)malloc(sizeof(int) * (rows+1));
        h_csrColInd = (int *)malloc(sizeof(int) * nnz);

        cudaStat1 = cudaMemcpy(h_csrvalues, d_csrvalues, sizeof(double)*nnz, cudaMemcpyDeviceToHost);
        cudaStat2 = cudaMemcpy(h_csrRowPtr, d_csrRowPtr, sizeof(int)*(rows+1), cudaMemcpyDeviceToHost);
        cudaStat3 = cudaMemcpy(h_csrColInd, d_csrColInd, sizeof(int)*nnz, cudaMemcpyDeviceToHost);
        assert(cudaSuccess == cudaStat1);
        assert(cudaSuccess == cudaStat2);
        assert(cudaSuccess == cudaStat3);
    */
        cudaStat1 = cudaDeviceSynchronize();     
        assert(cudaSuccess == cudaStat1);   

        //cudaStat1 = cudaMemcpy(h_cscVal, d_cscVal, sizeof(double)*nnz, cudaMemcpyDeviceToHost);
        //Print statement to see if it works. 
        /*
           printf("After CSC to CSR\n");
           for(int j = 0 ; j < nnz; j++){
           printf("(%d, %d, %f) \n", h_csrColInd[j], h_csrRowPtr[h_csrColInd[j]], h_csrvalues[j] );
           }
         */

        //cusolver and cusparse are different libraries and need a different handle.
        cusolverSpHandle_t solhandle = NULL;
        cusolverSpCreate(&solhandle);

        //Allocate for b of Ax=b on the device.
        double * d_b = NULL;
        cudaStat1 = cudaMalloc((void**)&d_b, sizeof(double)*rows);
        assert(cudaSuccess == cudaStat1);

        //Required for QR solver
        double tol = 1.e-12;
        const int reorder = 1; /*  reordering */
        int singularity = 0; /* -1 if A is invertible under tol. */

        //Allocate for x of Ax=b on the device
        double * d_x = NULL;
        cudaStat1 = cudaMalloc((void**)&d_x, sizeof(double)*rows);
        assert(cudaSuccess == cudaStat1);   

        //copy values from host to device
        cudaStat1 = cudaMemcpy(d_b, bpass, sizeof(double)*cols, cudaMemcpyHostToDevice);

        cudaDeviceSynchronize();//For timing


        cusolverStatus_t status1 = CUSOLVER_STATUS_SUCCESS; 
        gettimeofday(&t1, 0);
        status1 = cusolverSpDcsrlsvqr(
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
                &singularity);


        cudaDeviceSynchronize();//for timing
        gettimeofday(&t2, 0);

        if (0 <= singularity)
        {
            printf("WARNING: the matrix is singular at row %d under tol (%E)\n", singularity, tol);
        }

        assert(cudaSuccess == status1);   

        double time = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1e6;
        printf("Time to solve:  %3.1f s \n", time);

        xreturn   = (double*)malloc(sizeof(double)*cols);
        cudaStat1 = cudaMemcpy(xreturn, d_x, sizeof(double)*cols, cudaMemcpyDeviceToHost);  

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
