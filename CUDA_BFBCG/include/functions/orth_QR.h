#ifndef ORTH_QR_H
#define ORTH_QR_H


#include <iostream>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cublas_v2.h>
#include <cstdlib>
#include <cmath>
#include <sys/time.h>
#include <magma_v2.h>
#include <magma_lapack.h>

// helper function CUDA error checking and initialization
#include "helper.h"
#include "cuBLAS_util.h"
#include "cuSOLVER_util.h"

//Input: double* mtxQ_trnc_d, double* mtxZ_d, int number of Row, int number Of column, int & currentRank
//Process: the function extracts orthonormal set from the matrix Z
//Output: double* mtxQ_trnc_d, the orthonormal set of matrix Z with significant column vectors
void orth_QR(double** mtxQ_trnc_d, double* mtxZ_d, int numOfRow, int numOfClm, int &currentRank);

//Input: singluar values, int currnet rank, double threashould
//Process: Check eigenvalues are greater than threashould, then set new rank
//Output: int newRank
int setRank(double* sngVals_d, int currentRank, double threashold);





void orth_QR(double** mtxQ_trnc_d, double* mtxZ_d, int numOfRow, int numOfClm, int &currentRank)
{
    magma_init();

    const double THREASHOLD = 1e-6;

    double *mtxZ_cpy_d = NULL;
    double *mtxQ_d = NULL;  
    magma_int_t lda = numOfRow; // leading dimenstion
    magma_int_t lwork = 0;
    magma_int_t nb = magma_get_dgeqrf_nb(numOfRow, numOfClm); // Block size
    magma_int_t info = 0;
    magma_int_t* ipiv = NULL; //pivotig information
    double *tau_d = NULL;
    double *dT = NULL;
    double *work_d = NULL;

    

    //(1) Allocate memory
    CHECK(cudaMalloc((void**)&mtxZ_cpy_d, numOfRow * numOfClm * sizeof(double)));
    CHECK(cudaMalloc((void**)&ipiv, numOfClm * sizeof(magma_int_t)));
    CHECK(cudaMalloc((void**)&tau_d, numOfClm * sizeof(double)));
    CHECK(cudaMalloc((void**)&dT, (2 * nb * numOfRow + (2*nb+1) * nb) * sizeof(double)));

    //(2) Copy value to device
    CHECK(cudaMemcpy(mtxZ_cpy_d, mtxZ_d, numOfRow * numOfClm * sizeof(double), cudaMemcpyDeviceToDevice));

    //(3) Calculate lwork based on the documentation
    lwork = (numOfClm + 1) * nb + 2 * numOfClm;

    //(4) Allocate workspace 
    CHECK(cudaMalloc((void**)&work_d, lwork * sizeof(double)));

    printf("\nBefore magma_dgeqp3_gpu\n");
    //(5) Perform QR decompostion with column pivoting
    //mtxZ_cpy_d contains the R matrix in its upper triangular part,
    //the orthogonal matrix Q is represented implicitly by the Householder vectors stored in the lower triangular part of mtxZ_cpy_d and in tau_d, scalor.
    CHECK_MAGMA(magma_dgeqp3_gpu((magma_int_t)numOfRow, (magma_int_t)numOfClm, mtxZ_cpy_d, lda, ipiv, tau_d, work_d, lwork, &info));

    //(6) Iterate diagonal elements matrix R that is upper trianguar part of mtxZ_cpy_d to find rank
    currentRank = setRank(mtxZ_cpy_d, currentRank, THREASHOLD);


    //(7) Create an identity matrix on the device to extract matrix Q
    CHECK_MAGMA(magma_dorgqr_gpu((magma_int_t)numOfRow, (magma_int_t)numOfRow, (magma_int_t)numOfClm, mtxZ_cpy_d, lda, tau_d, dT, nb, &info));


    //(8) Copy matrix Q to the mtxQ_trnc_d with significant column vectors
    CHECK(cudaMemcpy(mtxQ_trnc_d, mtxZ_cpy_d, numOfRow * currentRank * sizeof(double), cudaMemcpyDeviceToDevice));

    //(9) Clean up
    CHECK(cudaFree(mtxZ_cpy_d));
    CHECK(cudaFree(ipiv));
    CHECK(cudaFree(tau_d));
    CHECK(cudaFree(work_d));
    CHECK(cudaFree(dT));

    magma_finalize();

} // end of orth_QR


//Input: singluar values, int currnet rank, double threashould
//Process: Check eigenvalues are greater than threashould, then set new rank
//Output: int newRank
int setRank(double* sngVals_d, int currentRank, double threashold)
{   
    int newRank = 0;

    //Allcoate in heap to copy value from device
    double* sngVals_h = (double*)malloc(currentRank * sizeof(double));
    // Copy singular values from Device to count eigen values
    CHECK(cudaMemcpy(sngVals_h, sngVals_d, currentRank * sizeof(double), cudaMemcpyDeviceToHost));

    for(int wkr = 0; wkr < currentRank; wkr++){
        if(sngVals_h[wkr] > threashold){
            newRank++;
        } // end of if
    } // end of for

    free(sngVals_h);

    return newRank;
}




#endif // ORTH_QR_H