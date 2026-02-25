/* loadmimeticmat.cpp
 *
 * Compare Armadillo+SuperLU with CUDA+cuSPARSE for solving Ax=b where A is a
 * VERY sparse matrix. This version LOADS a premade mimetic operator matrix,
 * which is the representation of a 102x10x52 domain.
 * L is 53040x53040 with 1081067 nnz, L is not diagonal, pos. def., nor symmetric.
 * b is 53040x1 with 53040 nnz.
 *
 * C670 Project
 * This section just demonstrates how to pass the sp_mat to the CUDA function.
 * I just used a simple wrapper. I left some of the toy example sections in,
 * just commented out, in case any one needs to modify the code. But this is a
 * stable build. Other files required are cudaarmwrappers.h, the header file,
 * and gpusolve.cu where the function solveviacuda() is actually defined.
 *
 * This was borrowed from a SDSU Math 670 student. If you were this student,
 * please let me know so I can give you credit. I don't remember your name,
 * sorry!
 *
 * Created ****** April 2024
 * Modified: Jared Brzenski February 2026
 *
 **/

#include <armadillo>
#include <chrono>
#include <iostream>

#include "cudaarmwrappers.h"

using namespace std;
using namespace arma;
using namespace std::chrono;

int main() {
  // Define A and b of Ax=b
  cout << "" << endl;
  cout << "**********************************************************" << endl;
  cout << "Loading mimetic Laplacian (A)..." << endl;
  cout << " Original condition number of A is ~ inf " << endl;
  cout << "**********************************************************" << endl;

  sp_mat A;
  A.load("L_52x10x22_L.txt", arma::coord_ascii);
  cout << "Size of A: " << A.n_rows << " x " << A.n_cols << endl;
  cout << "Number of nonzeros in A: " << A.n_nonzero << endl;

  int size = A.n_rows; // Assuming A is square, this is also the size of b and x

  cout << "" << endl;
  cout << "**********************************************************" << endl;
  cout << "Loading rhs B vector..." << endl;
  cout << "**********************************************************" << endl;

  vec b;
  b.load("b_52x10x22_b.txt", arma::raw_ascii);
  cout << "Size of b: " << b.n_rows << " x " << b.n_cols << endl;

  /* Super LU First solve */
  cout << "" << endl;
  cout << "**********************************************************" << endl;
  cout << "Doing SuperLU uncoditioned solve..." << endl;
  cout << "**********************************************************" << endl;
  vec x1;

  /*struct superlu_opts
  {
  bool             allow_ugly;   // default: false
  bool             equilibrate;  // default: false
  bool             symmetric;    // default: false
  double           pivot_thresh; // default: 1.0
  permutation_type permutation;  // default: superlu_opts::COLAMD
  refine_type      refine;       // default: superlu_opts::REF_NONE
  };
  */
  // Set up SuperLU options. These have been tested as fastest for this LHS
  superlu_opts opts;
  opts.pivot_thresh = 0.0; // not for 4th order
  opts.permutation = superlu_opts::NATURAL; // 4 seconds

  // Precondition L matrix with diagonal scaling.
  sp_mat A_scaled = A;
  A_scaled += speye( arma::size( A ) ) * 1e-6;
  
  auto start = high_resolution_clock::now();
  spsolve(x1, A, b, "superlu",  opts);
  auto stop = high_resolution_clock::now();
  auto duration = duration_cast<microseconds>(stop - start);
  cout << "Time taken by SuperLU:" << duration.count() / 1e6 << " seconds"
       << endl;

  cout << "" << endl;
  cout << "**********************************************************" << endl;
  cout << "Doing SuperLU coditioned solve..." << endl;
  cout << "**********************************************************" << endl;

  start = high_resolution_clock::now();
  spsolve(x1, A_scaled, b, "superlu",  opts);
  stop = high_resolution_clock::now();
  duration = duration_cast<microseconds>(stop - start);
  cout << "Time taken by SuperLU:" << duration.count() / 1e6 << " seconds"
       << endl;
  cout << "" << endl;

  cout << "**********************************************************" << endl;
  cout << "Doing SuperLU Factorization and Solve w/ conditioned L..." << endl;
  cout << "**********************************************************" << endl;
  vec x3;
  spsolve_factoriser SF;

  start = high_resolution_clock::now();
  bool status = SF.factorise(A_scaled);
  if (status == false) {
    cout << "Factorization failed. Exiting." << endl;
      return -1;
  }
  stop = high_resolution_clock::now();
  duration = duration_cast<microseconds>(stop - start);
  cout << "Time taken by factorization:" << duration.count() / 1e6 << " seconds"
       << endl;

  double rcond_value = SF.rcond();
  cout << "Reciprocal of condition number: " << rcond_value << endl;
  start = high_resolution_clock::now();
  bool solution_status = SF.solve(x3, b);
  stop = high_resolution_clock::now();
  duration = duration_cast<microseconds>(stop - start);
  if (solution_status == false) {
    cout << "couldn't find X3" << endl;
  }
  cout << "Time taken by SuperLU factored:" << duration.count() / 1e6
      << " seconds" << endl;

  vec x4;
  start = high_resolution_clock::now();
  solution_status = SF.solve(x4, b);
  stop = high_resolution_clock::now();
  duration = duration_cast<microseconds>(stop - start);
  if (solution_status == false) {
    cout << "couldn't find X4" << endl;
  }
  cout << "Time taken by SuperLU factored second time:"
       << duration.count() / 1e6 << " seconds" << endl;

  vec result;
  result = x1 - x3;
  double normof2arms = norm(result, 1);
  cout << "L1 Norm of OG and scaled SuperLU solutions:" << normof2arms << endl;

  // Grab all the elements of A, A is stored in CSC format.
  const double *values = A.values;
  const long long unsigned int nnz = A.n_nonzero;
  const long long unsigned int rows = A.n_rows;
  const long long unsigned int cols = A.n_cols;
  const long long unsigned int *row_indices = A.row_indices;
  const long long unsigned int *col_ptrs = A.col_ptrs;
  double *bpass = b.memptr(); // Convert B  Vec to standard array of doubles
  double *xreturn = NULL;

  cout << "" << endl;
  cout << "**********************************************************" << endl;
  cout << "Doing cusparse solve..." << endl;
  cout << "**********************************************************" << endl;

  start = high_resolution_clock::now();
  xreturn = Wrapper::solveviacuda(values, nnz, rows, cols, row_indices,
                                  col_ptrs, bpass, xreturn);
  stop = high_resolution_clock::now();

  duration = duration_cast<microseconds>(stop - start);
  cout << "Total time take from ENTIRE CUDA function call:"
       << duration.count() / 1e6 << " seconds" << endl;

  vec rfc(&xreturn[0], size); // THIS IS HOW YOU SHOVE A DOUBLE ARRAY BACK INTO
                              // AN ARMADILLO VEC CLASS
  result = x4 - rfc;
  double normavc = norm(result, 1);
  cout << "L1 Norm of Armadillo compared with Cuda computations:" << normavc
       << endl;
       
  // Relative residual: ||A*x - b|| / ||b||
  // Values near machine epsilon (~1e-15) are excellent.
  // Values above ~1e-6 suggest the solver struggled (ill-conditioned matrix).
  // Values near 1e-9 are good enough for modeling.
  double bnorm = norm(b, 2);
  vec residual_superlu = A * x1 - b;
  cout << "SuperLU   relative residual ||Ax-b||/||b||: "
       << norm(residual_superlu, 2) / bnorm << endl;

  vec residual_cuda = A * rfc - b;
  double rel_res_cuda = norm(residual_cuda, 2) / bnorm;
  cout << "CUDA      relative residual ||Ax-b||/||b||: " << rel_res_cuda << endl;

  if (rel_res_cuda > 1e-4) {
    cout << "WARNING: CUDA solution relative residual > 1e-4 "
            "(matrix may be ill-conditioned for LU)" << endl;
  }

  return 0;
}
