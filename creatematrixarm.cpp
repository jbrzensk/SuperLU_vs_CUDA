/* Compare Armadillo+SuperLU with CUDA+cuSPARSE for solving Ax=b where A is a 
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
 
#include <iostream>
#include <armadillo>
#include "cusparse.h"
#include "cudaarmwrappers.h"
#include <chrono>

using namespace std;
using namespace arma;
using namespace std::chrono;

int main()
{
    //Define A and b of Ax=b

    int size = 8192;
    sp_mat A = sprandn<sp_mat>(size, size, 0.03);
    vec b(size,      fill::randu);   

    //Toy exampl to ensure working
    /* 
       mat Adense = { { 0.0, 4.0, 7.0 },
       { 2.0, 5.0, 8.0 },
       { 3.0, 6.0, 10.0 } };

       b = {1.0,2.0,3.0};
     */
    mat Adense(A);

    mat Q;
    mat R;
    auto start = high_resolution_clock::now();
    qr(Q, R, Adense);
    vec x = R.i()*(Q.t()*b);
    auto stop = high_resolution_clock::now();

    auto duration = duration_cast<microseconds>(stop - start);
    cout << "Time taken by Armadillo QR:" << duration.count()/1e6 << " seconds" << endl;  

    //Toy example code
    //Adense.print("ARMA::Adense:");
    //x.print("ARMA:QR:x:");

    start = high_resolution_clock::now();
    vec x1 = solve(Adense, b);
    stop = high_resolution_clock::now();

    duration = duration_cast<microseconds>(stop - start);
    cout << "Time taken by Armadillo solver:" << duration.count()/1e6 << " seconds" << endl;  

    //Toy example sections
    //b.print("arma::B:");
    //x.print("ARMA::QRsolve::x:");
    //x1.print("ARMA::solve::x1:");

    vec result;
    result = x1-x;
    double normof2arms = norm(result,1);
    cout << "L1 Norm of both the armadillo computations:" << normof2arms << endl;


    //Toy example: convert dense matrix to sparse
    //sp_mat A(Adense);    

    /*
     *
     *IF YOU WANT TO JUST USE THE CODE YOU CAN PRETTY MUCH IGNORE THE REST OF THIS FILE.
     *BETWEEN THIS COMMENT BLOCK AND THE NEXT IS THE CODE TO USE THE FUNCTION, PLUS THE TIMING CODE.
     *ALL YOU NEED ARE AN SP_MAT A, VEC B, AND A DOUBLE POINTER FOR X.
     *
     */
    //Grab all the elements of A, A is stored in CSC format. 
    const double * values = A.values;
    const long long unsigned int nnz = A.n_nonzero;
    const long long unsigned int rows = A.n_rows;
    const long long unsigned int cols = A.n_cols;
    const long long unsigned int * row_indices = A.row_indices;
    const long long unsigned int * col_ptrs = A.col_ptrs;
    double *bpass = b.memptr();//Convert B from Armadillo Vec to standard array of doubles
    double * xreturn = NULL;

    start = high_resolution_clock::now();
    xreturn = Wrapper::solveviacuda(values, nnz, rows, cols, row_indices, col_ptrs, bpass, xreturn);   
    stop = high_resolution_clock::now();
    /*
     *I KNOW YOU HATE COMMENTS SO IF YOU WANT JUST READ BETWEEN THESE TWO BLOCKS. THE REST OF THIS
     *FILE IS PRETTY BOILER PLATE. 
     *
     */
    duration = duration_cast<microseconds>(stop - start);
    cout << "Total time take from CUDA function call:" << duration.count()/1e6 << " seconds" << endl;  


    vec rfc(&xreturn[0], size);//THIS IS HOW YOU SHOVE A DOUBLE ARRAY BACK INTO AN ARMADILLO VEC CLASS
    result = x1-rfc;
    double normavc = norm(result,1);
    cout << "L1 Norm of Armadillo compared with Cuda computations:" << normavc << endl;


    /* MORE TOY EXAMPLE TROUBLE SHOOTING CODE */
    //vec returnedfromcuda(3);
    //returnedfromcuda.memptr()=xreturn;
    //rfc.print("Final Print?");
    //Print Statements to compare results

    /*
       cout << "XRETURN FROM ARMA" << endl;
       for(int j = 0 ; j < 3; j++){
       printf("( %f) \n",  xreturn[j] );
       }
     */

    /*

       A.print("A in ARMA:");
       for(int j = 0 ; j < 3; j++){
       printf("( %f) \n",  bpass[j] );
       }

       printf("IN ARMA \n");
       for(int j = 0 ; j < nnz; j++){
       printf("(%llu, %llu, %f) \n", row_indices[j], col_ptrs[row_indices[j]],  values[j] );
       }
     */
    return 0;
}
