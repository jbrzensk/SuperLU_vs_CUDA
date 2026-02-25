/* creatematrixarm.cpp
 *
 * Compare Armadillo+SuperLU with CUDA+cuSPARSE for solving Ax=b where A is a
 * VERY sparse matrix.
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

  int size = 4096;
  cout << "**********************************************************" << endl;
  cout << "Creating a random sparse matrix of size " << size << "x" << size
       << " with 3% density." << endl;
  cout << "**********************************************************" << endl;

  sp_mat A = sprandn<sp_mat>(size, size, 0.03);
  vec b(size, fill::randu);

  mat Adense(A);

  mat Q;
  mat R;

  /* Armadillo QR Decomposition Solve */
  cout << "" << endl;
  cout << "**********************************************************" << endl;
  cout << "Doing QR solve (DENSE)..." << endl;
  cout << "**********************************************************" << endl;

  auto start = high_resolution_clock::now();
  qr(Q, R, Adense);
  vec x = R.i() * (Q.t() * b);
  auto stop = high_resolution_clock::now();

  auto duration = duration_cast<microseconds>(stop - start);
  cout << "Time taken by Armadillo QR solve:" << duration.count() / 1e6
       << " seconds" << endl;

  /* Super LU First solve */
  cout << "" << endl;
  cout << "**********************************************************" << endl;
  cout << "Doing SuperLU solve..." << endl;
  cout << "**********************************************************" << endl;
  vec x1;
  start = high_resolution_clock::now();
  spsolve(x1, A, b);
  stop = high_resolution_clock::now();
  duration = duration_cast<microseconds>(stop - start);
  cout << "Time taken by SuperLU:" << duration.count() / 1e6 << " seconds"
       << endl;

  cout << "" << endl;
  cout << "**********************************************************" << endl;
  cout << "Doing SuperLU Factorization and Solve..." << endl;
  cout << "**********************************************************" << endl;
  vec x3;
  spsolve_factoriser SF;

  start = high_resolution_clock::now();
  bool status = SF.factorise(A);
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
  cout << "Time taken by SuperLU factored secodn time:"
       << duration.count() / 1e6 << " seconds" << endl;

  vec result;
  result = x1 - x;
  double normof2arms = norm(result, 1);
  cout << "L1 Norm of QR and SuperLU solutions:" << normof2arms << endl;

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
  result = x1 - rfc;
  double normavc = norm(result, 1);
  cout << "L1 Norm of Armadillo compared with Cuda computations:" << normavc
       << endl;

  return 0;
}
