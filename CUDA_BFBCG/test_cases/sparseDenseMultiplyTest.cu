// includes, system
#include<stdio.h>
#include<stdlib.h>
#include<string.h>

/*Using updated (v2) interfaces to cublas*/
#include<cublas_v2.h>
#include<cuda_runtime.h>
#include<cusparse.h>
#include <cusolverDn.h>
#include<sys/time.h>


//Utilities
#include "../includes/helper_debug.h"
// helper function CUDA error checking and initialization
#include "../includes/helper_cuda.h"  
#include "../includes/helper_functions.h"
#include "../includes/cusolver_utils.h"
#include "../includes/CSRMatrix.h"


#define CHECK(call){ \
    const cudaError_t cuda_ret = call; \
    if(cuda_ret != cudaSuccess){ \
        printf("Error: %s:%d,  ", __FILE__, __LINE__ );\
        printf("code: %d, reason: %s \n", cuda_ret, cudaGetErrorString(cuda_ret));\
        exit(-1); \
    }\
}


void sparseDenseMultiplyTest_Case1();
void sparseDenseMultiplyTest_Case2();
void sparseDenseMultiplyTest_Case3();
void sparseDenseMultiplyTest_Case4();
void sparseDenseMultiplyTest_Case5();

int main(int arg, char** argv)
{
    
    printf("\n\n= = = =sparseDenseMultiplyTest.cu= = = = \n\n");
    
    printf("\n\n🔍🔍🔍 Test Case 1 🔍🔍🔍\n\n");
    sparseDenseMultiplyTest_Case1();

    printf("\n\n🔍🔍🔍 Test Case 2 🔍🔍🔍\n\n");
    sparseDenseMultiplyTest_Case2();

    printf("\n\n🔍🔍🔍 Test Case 3 🔍🔍🔍\n\n");
    sparseDenseMultiplyTest_Case3();

    // printf("\n\n🔍🔍🔍 Test Case 4 🔍🔍🔍\n\n");
    // sparseDenseMultiplyTest_Case4();

    // printf("\n\n🔍🔍🔍 Test Case 5 🔍🔍🔍\n\n");
    // sparseDenseMultiplyTest_Case5();

    printf("\n\n= = = = ✅end of sparseDenseMultiplyTest✅ = = = =\n\n");

    return 0;
} // end of main




void sparseDenseMultiplyTest_Case1()
{
    // Number of matrix A rows and columns
    const int N = 5;


    // Define the dense matrixB column major
    double dnsMtxB_h[] = {
        0.1, 0.2, 0.3, 0.4, 0.5,
        0.6, 0.7, 0.8, 0.9, 1.0,
        1.1, 1.2, 1.3, 1.4, 1.5
    };

    int numOfRowsB = 5;
    int numOfClmsB = 3;

    double* dnsMtxB_d = NULL;
    double* dnsMtxC_d = NULL;

    bool debug = true;



    //Generate sparse Identity matrix
    CSRMatrix csrMtxI_h = generateSparseIdentityMatrixCSR(N);


    if(debug){
        
        //Sparse matrix information to check
        printf("\n\n~~mtxI sparse~~\n\n");
        print_CSRMtx(csrMtxI_h);
        
        //Converst CSR to dense matrix to check
        double *dnsMtx = csrToDense(csrMtxI_h);
        double *mtxI_d = NULL;
        CHECK(cudaMalloc((void**)&mtxI_d, N * N * sizeof(double)));
        CHECK(cudaMemcpy(mtxI_d, dnsMtx, N * N * sizeof(double), cudaMemcpyHostToDevice));
        
        printf("\n\n~~mtxI dense~~\n\n");
        print_mtx_clm_d(mtxI_d, N, N);
        
        CHECK(cudaFree(mtxI_d));
        free(dnsMtx);
    }


    //(1) Allocate memeory 
    CHECK(cudaMalloc((void**)&dnsMtxB_d, numOfRowsB * numOfClmsB * sizeof(double)));
    CHECK(cudaMalloc((void**)&dnsMtxC_d, N * numOfClmsB * sizeof(double)));

    //(2) Copy values from host to device
    CHECK(cudaMemcpy(dnsMtxB_d, dnsMtxB_h, numOfRowsB * numOfClmsB * sizeof(double), cudaMemcpyHostToDevice));

    if(debug){
        //Sparse matrix information to check
        printf("\n\n~~dense matrix B~~\n\n");
        print_mtx_clm_d(dnsMtxB_d, numOfRowsB, numOfClmsB);
    }

    //Call sparseMultiplySense function
    multiply_Sprc_Den_mtx(csrMtxI_h, dnsMtxB_d, numOfClmsB, dnsMtxC_d);

    //Get dense matrix as a result
    printf("\n\n~~dense matrix C after multiplication~~\n\n");
    print_mtx_clm_d(dnsMtxC_d, N, numOfClmsB);


    //Free memory
    CHECK(cudaFree(dnsMtxB_d));
    CHECK(cudaFree(dnsMtxC_d));
    freeCSRMatrix(csrMtxI_h);



} // end of sparseDenseMultiplyTest_Case1




void sparseDenseMultiplyTest_Case2()
{
    // Number of matrix A rows and columns
    const int N = 5;

    // Define the dense matrixB column major
    double dnsMtxB_h[] = {
        0.1, 0.2, 0.3, 0.4, 0.5,
        0.6, 0.7, 0.8, 0.9, 1.0,
        1.1, 1.2, 1.3, 1.4, 1.5
    };

    int rowOffSets[] = {0, 2, 5, 8, 11, 13};
    int colIndices[] = {0, 1, 0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4};
    double vals[] = {10.840188, 0.394383, 
                    0.394383, 10.783099, 0.798440, 
                    0.798440, 10.911648, 0.197551,
                    0.197551, 10.335223, 0.768230,
                    0.768230, 10.277775};


    int numOfRowsB = 5;
    int numOfClmsB = 3;

    double* dnsMtxB_d = NULL;
    double* dnsMtxC_d = NULL;

    // bool debug = true;



    //Generate sparse Identity matrix
    CSRMatrix csrMtxA_h;

    csrMtxA_h.numOfRows = N;
    csrMtxA_h.numOfClms = N;
    csrMtxA_h.numOfnz = 13;
    csrMtxA_h.row_offsets = rowOffSets;
    csrMtxA_h.col_indices = colIndices;
    csrMtxA_h.vals = vals;


    
    //Sparse matrix information to check
    printf("\n = = = Before sparse multiplication operation = = = \n\n");
    printf("\n\n~~mtxA sparse~~\n");
    print_CSRMtx(csrMtxA_h);
    
    //Converst CSR to dense matrix to check
    double* dnsMtx = csrToDense(csrMtxA_h);
    double *mtxA_d = NULL;
    CHECK(cudaMalloc((void**)&mtxA_d, N * N * sizeof(double)));
    CHECK(cudaMemcpy(mtxA_d, dnsMtx, N * N * sizeof(double), cudaMemcpyHostToDevice));
    
    printf("\n\n~~mtxA dense~~\n\n");
    print_mtx_clm_d(mtxA_d, N, N);
    


    //(1) Allocate memeory 
    CHECK(cudaMalloc((void**)&dnsMtxB_d, numOfRowsB * numOfClmsB * sizeof(double)));
    CHECK(cudaMalloc((void**)&dnsMtxC_d, N * numOfClmsB * sizeof(double)));

    //(2) Copy values from host to device
    CHECK(cudaMemcpy(dnsMtxB_d, dnsMtxB_h, numOfRowsB * numOfClmsB * sizeof(double), cudaMemcpyHostToDevice));

    //Sparse matrix information to check
    printf("\n\n~~dense matrix B~~\n\n");
    print_mtx_clm_d(dnsMtxB_d, numOfRowsB, numOfClmsB);
    


    //(3)Call sparseMultiplySense function
    multiply_Sprc_Den_mtx(csrMtxA_h, dnsMtxB_d, numOfClmsB, dnsMtxC_d);

    //Get dense matrix as a result
    printf("\n\n~~dense matrix C after multiplication~~\n\n");
    print_mtx_clm_d(dnsMtxC_d, N, numOfClmsB);

    
    //Sparse matrix information to check
    printf("\n = = = After sparse multiplication operation = = = \n\n");
    printf("\n\n~~mtxA sparse~~\n");
    print_CSRMtx(csrMtxA_h);
    

    printf("\n\n~~mtxA dense~~\n\n");
    print_mtx_clm_d(mtxA_d, N, N);
    
    //Sparse matrix information to check
    printf("\n\n~~dense matrix B~~\n\n");
    print_mtx_clm_d(dnsMtxB_d, numOfRowsB, numOfClmsB);


    //(4)Free memory
    CHECK(cudaFree(mtxA_d));
    free(dnsMtx);
    CHECK(cudaFree(dnsMtxB_d));
    CHECK(cudaFree(dnsMtxC_d));
    // freeCSRMatrix(csrMtxA_h);

} // end of sparseDenseMultiplyTest_Case2




void sparseDenseMultiplyTest_Case3()
{
    // Number of matrix A rows and columns
    const int N = 7;

    // Define the dense matrixA column major
    // double dnsMtxA_h[] = {        
    //     10.840188, 0.394383, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 
    //     0.394383, 10.783099, 0.798440, 0.000000, 0.000000, 0.000000, 0.000000, 
    //     0.000000, 0.798440, 10.911648, 0.197551, 0.000000, 0.000000, 0.000000, 
    //     0.000000, 0.000000, 0.197551, 10.335223, 0.768230, 0.000000, 0.000000, 
    //     0.000000, 0.000000, 0.000000, 0.768230, 10.277775, 0.553970, 0.000000, 
    //     0.000000, 0.000000, 0.000000, 0.000000, 0.553970, 10.477397, 0.628871, 
    //     0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.628871, 10.364784 
    // };

    int rowOffSets[] = {0, 2, 5, 8, 11, 14, 17, 19};
    int colIndices[] = {0, 1, 0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4, 5, 4, 5, 6, 5, 6};
    double vals[] = {10.840188, 0.394383,
                    0.394383, 10.783099, 0.798440,
                    0.798440, 10.911648, 0.197551, 
                    0.197551, 10.335223, 0.768230,
                    0.768230, 10.277775, 0.553970, 
                    0.553970, 10.477397, 0.628871,
                    0.628871, 10.364784};


    // Define the dense matrixB column major
    double dnsMtxB_h[] = {
        0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,
        0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 
        1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1,
        2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8
    };

    int numOfRowsB = 7;
    int numOfClmsB = 4;

    double* dnsMtxB_d = NULL;
    double* dnsMtxC_d = NULL;

    // bool debug = true;


    //Generate sparse Identity matrix
    CSRMatrix csrMtxA_h;

    csrMtxA_h.numOfRows = N;
    csrMtxA_h.numOfClms = N;
    csrMtxA_h.numOfnz = 19;
    csrMtxA_h.row_offsets = rowOffSets;
    csrMtxA_h.col_indices = colIndices;
    csrMtxA_h.vals = vals;

        
    //Sparse matrix information to check
    printf("\n = = = Before sparse multiplication operation = = = \n\n");
    printf("\n\n~~mtxA sparse~~\n\n");
    print_CSRMtx(csrMtxA_h);
    
    //Converst CSR to dense matrix to check
    double* dnsMtx = csrToDense(csrMtxA_h);
    double *mtxA_d = NULL;
    CHECK(cudaMalloc((void**)&mtxA_d, N * N * sizeof(double)));
    CHECK(cudaMemcpy(mtxA_d, dnsMtx, N * N * sizeof(double), cudaMemcpyHostToDevice));
    
    printf("\n\n~~mtxA dense~~\n\n");
    print_mtx_clm_d(mtxA_d, N, N);
    

    //(1) Allocate memeory 
    CHECK(cudaMalloc((void**)&dnsMtxB_d, numOfRowsB * numOfClmsB * sizeof(double)));
    CHECK(cudaMalloc((void**)&dnsMtxC_d, N * numOfClmsB * sizeof(double)));

    //(2) Copy values from host to device
    CHECK(cudaMemcpy(dnsMtxB_d, dnsMtxB_h, numOfRowsB * numOfClmsB * sizeof(double), cudaMemcpyHostToDevice));


    //Sparse matrix information to check
    printf("\n\n~~dense matrix B~~\n\n");
    print_mtx_clm_d(dnsMtxB_d, numOfRowsB, numOfClmsB);
    

    //Call sparseMultiplySense function
    multiply_Sprc_Den_mtx(csrMtxA_h, dnsMtxB_d, numOfClmsB, dnsMtxC_d);

    //Get dense matrix as a result
    printf("\n\n~~dense matrix C after multiplication~~\n\n");
    print_mtx_clm_d(dnsMtxC_d, N, numOfClmsB);


        
    //Sparse matrix information to check
    printf("\n = = = After sparse multiplication operation = = = \n\n");
    printf("\n\n~~mtxA sparse~~\n\n");
    print_CSRMtx(csrMtxA_h);
    

    printf("\n\n~~mtxA dense~~\n\n");
    print_mtx_clm_d(mtxA_d, N, N);
    
    //Sparse matrix information to check
    printf("\n\n~~dense matrix B~~\n\n");
    print_mtx_clm_d(dnsMtxB_d, numOfRowsB, numOfClmsB);


    //Free memory
    CHECK(cudaFree(mtxA_d));
    free(dnsMtx);
    CHECK(cudaFree(dnsMtxB_d));
    CHECK(cudaFree(dnsMtxC_d));
    // freeCSRMatrix(csrMtxA_h);

} // end of sparseDenseMultiplyTest_Case1




void sparseDenseMultiplyTest_Case4()
{

} // end of sparseDenseMultiplyTest_Case1




void sparseDenseMultiplyTest_Case5()
{

} // end of sparseDenseMultiplyTest_Case1