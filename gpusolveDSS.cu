/* gpusolveDSS.cu
 * Solves sparse Ax=b using cuDSS (CUDA Direct Sparse Solver, CUDA 12+).
 *
 * cuDSS performs all three phases entirely on the GPU:
 *   1. Symbolic analysis  - reordering, sparsity pattern of L and U factors
 *   2. Numerical factorization - LU decomposition
 *   3. Triangular solve   - forward/back substitution
 *
 * This replaces gpusolve.cu which used cusolverSpDcsrlsvqr, a hybrid
 * CPU+GPU solver that performed the actual factorization on the CPU.
 *
 * Link with: -lcudss -lcusparse -lcudart
 * Compile with: nvcc -arch=sm_89 -c gpusolveDSS.cu
 */
#include "cudaarmwrappers.h"
#include "cusparse.h"
#include <cuda_runtime.h>
#include <cudss.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t _err = (call);                                                 \
    if (_err != cudaSuccess) {                                                 \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,         \
              cudaGetErrorString(_err));                                       \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

#define CUSPARSE_CHECK(call)                                                   \
  do {                                                                         \
    cusparseStatus_t _err = (call);                                            \
    if (_err != CUSPARSE_STATUS_SUCCESS) {                                     \
      fprintf(stderr, "cuSPARSE error at %s:%d: %s\n", __FILE__, __LINE__,    \
              cusparseGetErrorString(_err));                                   \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

#define CUDSS_CHECK(call)                                                      \
  do {                                                                         \
    cudssStatus_t _err = (call);                                               \
    if (_err != CUDSS_STATUS_SUCCESS) {                                        \
      fprintf(stderr, "cuDSS error at %s:%d: %d\n", __FILE__, __LINE__,       \
              (int)_err);                                                      \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

namespace Wrapper {

double *solveviacuda(const double *values, const long long unsigned int a_nnz,
                     const long long unsigned int rows,
                     const long long unsigned int cols,
                     const long long unsigned int *row_ind,
                     const long long unsigned int *col_ptrs, double *bpass,
                     double *xreturn) {

  if (a_nnz == 0 || rows == 0 || cols == 0) {
    fprintf(stderr,
            "solveviacuda: invalid dimensions (nnz=%llu rows=%llu cols=%llu)\n",
            a_nnz, rows, cols);
    return nullptr;
  }

  struct timeval t1, t2;
  int nnz = (int)a_nnz;

  // Cast Armadillo's 64-bit index arrays to 32-bit int required by CUDA sparse APIs
  int *h_csccol_pts = (int *)malloc(sizeof(int) * (cols + 1));
  for (int i = 0; i < (int)(cols + 1); i++) {
    h_csccol_pts[i] = (int)col_ptrs[i];
  }

  int *h_cscRowInd = (int *)malloc(sizeof(int) * nnz);
  for (int i = 0; i < nnz; i++) {
    h_cscRowInd[i] = (int)row_ind[i];
  }

  // Temporary CSC device buffers (freed immediately after CSC->CSR conversion)
  double *d_cscVal     = NULL;
  int    *d_csccol_pts = NULL;
  int    *d_cscRowInd  = NULL;

  // CSR device buffers (kept for cuDSS)
  double *d_csrvalues = NULL;
  int    *d_csrRowPtr = NULL;
  int    *d_csrColInd = NULL;

  CUDA_CHECK(cudaMalloc((void **)&d_cscVal,     sizeof(double) * nnz));
  CUDA_CHECK(cudaMalloc((void **)&d_csccol_pts, sizeof(int)    * (cols + 1)));
  CUDA_CHECK(cudaMalloc((void **)&d_cscRowInd,  sizeof(int)    * nnz));
  CUDA_CHECK(cudaMalloc((void **)&d_csrvalues,  sizeof(double) * nnz));
  CUDA_CHECK(cudaMalloc((void **)&d_csrRowPtr,  sizeof(int)    * (rows + 1)));
  CUDA_CHECK(cudaMalloc((void **)&d_csrColInd,  sizeof(int)    * nnz));

  // Upload CSC data — copy values directly from Armadillo, no intermediate host buffer needed
  CUDA_CHECK(cudaMemcpy(d_cscVal,     values,       sizeof(double) * nnz,        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_csccol_pts, h_csccol_pts, sizeof(int)    * (cols + 1), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_cscRowInd,  h_cscRowInd,  sizeof(int)    * nnz,        cudaMemcpyHostToDevice));

  free(h_csccol_pts);
  free(h_cscRowInd);

  // Convert CSC -> CSR on GPU using cuSPARSE
  // cuDSS requires CSR; cusparseCsr2cscEx2 works for both directions
  cusparseHandle_t     sphandle   = NULL;
  cusparseIndexBase_t  idxBase    = CUSPARSE_INDEX_BASE_ZERO;
  cusparseAction_t     copyValues = CUSPARSE_ACTION_NUMERIC;
  cusparseCsr2CscAlg_t alg        = CUSPARSE_CSR2CSC_ALG1;
  cudaDataType         valType    = CUDA_R_64F;

  CUSPARSE_CHECK(cusparseCreate(&sphandle));

  // Query workspace size — host-side call, no sync needed before malloc
  size_t  bufferSize = 0;
  double *d_buffer   = NULL;
  CUSPARSE_CHECK(cusparseCsr2cscEx2_bufferSize(
      sphandle, cols, rows, nnz,
      d_cscVal, d_csccol_pts, d_cscRowInd,
      d_csrvalues, d_csrRowPtr, d_csrColInd,
      valType, copyValues, idxBase, alg, &bufferSize));

  CUDA_CHECK(cudaMalloc((void **)&d_buffer, bufferSize > 0 ? bufferSize : 1));

  CUSPARSE_CHECK(cusparseCsr2cscEx2(
      sphandle, cols, rows, nnz,
      d_cscVal, d_csccol_pts, d_cscRowInd,
      d_csrvalues, d_csrRowPtr, d_csrColInd,
      valType, copyValues, idxBase, alg, d_buffer));

  CUDA_CHECK(cudaDeviceSynchronize());

  // CSC buffers no longer needed — free immediately to reduce peak GPU memory
  cudaFree(d_cscVal);
  cudaFree(d_csccol_pts);
  cudaFree(d_cscRowInd);
  cudaFree(d_buffer);
  cusparseDestroy(sphandle);

  // Allocate b and x on device
  double *d_b = NULL;
  double *d_x = NULL;
  CUDA_CHECK(cudaMalloc((void **)&d_b, sizeof(double) * rows));
  CUDA_CHECK(cudaMalloc((void **)&d_x, sizeof(double) * rows));
  CUDA_CHECK(cudaMemcpy(d_b, bpass, sizeof(double) * rows, cudaMemcpyHostToDevice));

  // cuDSS setup
  cudssHandle_t dssHandle = NULL;
  cudssConfig_t dssConfig = NULL;
  cudssData_t   dssData   = NULL;
  CUDSS_CHECK(cudssCreate(&dssHandle));
  CUDSS_CHECK(cudssConfigCreate(&dssConfig));
  CUDSS_CHECK(cudssDataCreate(dssHandle, &dssData));

  // Describe A as a general sparse matrix in CSR format
  cudssMatrix_t matA = NULL;
  CUDSS_CHECK(cudssMatrixCreateCsr(
      &matA,
      (int64_t)rows, (int64_t)cols, (int64_t)nnz,
      d_csrRowPtr, NULL,   // NULL rowEnd = standard CSR (not CSR3)
      d_csrColInd,
      d_csrvalues,
      CUDA_R_32I,          // index type (row pointer and column index)
      CUDA_R_64F,          // value type
      CUDSS_MTYPE_GENERAL, // matrix type
      CUDSS_MVIEW_FULL,    // matrix view
      CUDSS_BASE_ZERO));   // index base

  // Describe b and x as dense column vectors
  cudssMatrix_t matB = NULL, matX = NULL;
  CUDSS_CHECK(cudssMatrixCreateDn(
      &matB, (int64_t)rows, 1, (int64_t)rows,
      d_b, CUDA_R_64F, CUDSS_LAYOUT_COL_MAJOR));
  CUDSS_CHECK(cudssMatrixCreateDn(
      &matX, (int64_t)rows, 1, (int64_t)rows,
      d_x, CUDA_R_64F, CUDSS_LAYOUT_COL_MAJOR));

  // Three-phase GPU-side solve
  cudaDeviceSynchronize();
  gettimeofday(&t1, 0);

  // Phase 1: symbolic analysis (reordering, sparsity of factors)
  CUDSS_CHECK(cudssExecute(dssHandle, CUDSS_PHASE_ANALYSIS,
                           dssConfig, dssData, matA, matX, matB));

  // Phase 2: numerical factorization (LU decomposition on GPU)
  CUDSS_CHECK(cudssExecute(dssHandle, CUDSS_PHASE_FACTORIZATION,
                           dssConfig, dssData, matA, matX, matB));

  // Phase 3: triangular solve (forward/back substitution on GPU)
  CUDSS_CHECK(cudssExecute(dssHandle, CUDSS_PHASE_SOLVE,
                           dssConfig, dssData, matA, matX, matB));

  cudaDeviceSynchronize();
  gettimeofday(&t2, 0);

  double time =
      (1000000.0 * (t2.tv_sec - t1.tv_sec) + t2.tv_usec - t1.tv_usec) / 1e6;
  printf("cuDSS time to solve: %3.3f s\n", time);

  xreturn = (double *)malloc(sizeof(double) * cols);
  CUDA_CHECK(cudaMemcpy(xreturn, d_x, sizeof(double) * cols, cudaMemcpyDeviceToHost));

  // Cleanup cuDSS
  cudssMatrixDestroy(matA);
  cudssMatrixDestroy(matB);
  cudssMatrixDestroy(matX);
  cudssDataDestroy(dssHandle, dssData);
  cudssConfigDestroy(dssConfig);
  cudssDestroy(dssHandle);

  // Cleanup CUDA
  cudaFree(d_csrvalues);
  cudaFree(d_csrRowPtr);
  cudaFree(d_csrColInd);
  cudaFree(d_x);
  cudaFree(d_b);

  return xreturn;
}

} // namespace Wrapper
