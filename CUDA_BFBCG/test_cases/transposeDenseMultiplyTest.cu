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


void transposeDenseMultiplyTest_Case1();
void transposeDenseMultiplyTest_Case2();
void transposeDenseMultiplyTest_Case3();
void transposeDenseMultiply_Case4();
void transposeDenseMultiply_Case5();

int main(int arg, char** argv)
{
    
    printf("\n\n= = = =tranposeTest.cu= = = = \n\n");
    
    printf("\n\n🔍🔍🔍 Test Case 1 🔍🔍🔍\n\n");
    transposeDenseMultiplyTest_Case1();

    printf("\n\n🔍🔍🔍 Test Case 2 🔍🔍🔍\n\n");
    transposeDenseMultiplyTest_Case2();

    printf("\n\n🔍🔍🔍 Test Case 3 🔍🔍🔍\n\n");
    transposeDenseMultiplyTest_Case3();

    // printf("\n\n🔍🔍🔍 Test Case 4 🔍🔍🔍\n\n");
    // transposeDenseMultiplyTest_Case4();

    // printf("\n\n🔍🔍🔍 Test Case 5 🔍🔍🔍\n\n");
    // transposeDenseMultiplyTest_Case5();

    printf("\n\n= = = = ✅end of tranposeTest✅ = = = =\n\n");

    return 0;
} // end of main




void transposeDenseMultiplyTest_Case1()
{
    const int M = 3;
    const int K = 2;
    const int N = 2;

    double mtxA[] = {4.0, 1.0, 1.0, 
                    1.0, 3.0, 1.0};

    double mtxB[] = {1.0, 2.0, 3.0, 
                    4.0, 5.0, 6.0};
    double* mtxA_d = NULL;
    double* mtxB_d = NULL;
    double* mtxC_d = NULL;


    int numOfRowA = M;
    int numOfColA = K;
    int numOfRowB = M;
    int numOfColB = N;
    int numOfRowC = K;
    int numOfColC = N;

    bool debug = true;


    //(1) Allocate device memeory
    CHECK(cudaMalloc((void**)&mtxA_d, numOfRowA * numOfColA * sizeof(double)));
    CHECK(cudaMalloc((void**)&mtxB_d, numOfRowB * numOfColB * sizeof(double)));
    CHECK(cudaMalloc((void**)&mtxC_d, numOfRowC * numOfColC * sizeof(double)));

    //(2) Copy data to device
    CHECK(cudaMemcpy(mtxA_d, mtxA, numOfRowA * numOfColA * sizeof(double), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(mtxB_d, mtxB, numOfRowB * numOfColB * sizeof(double), cudaMemcpyHostToDevice));

    if(debug){
        printf("\n\n = = = Before transpose operation = = =\n");
        printf("\n\n~~mtxA_d~~\n\n");
        print_mtx_clm_d(mtxA_d, numOfRowA, numOfColA);
        printf("\n\n~~mtxB_d~~\n\n");
        print_mtx_clm_d(mtxB_d, numOfRowB, numOfColB);
    }


    //(4) mtxC <- mtxA' * mtxB
    cublasHandle_t cublasHandler = NULL;
    checkCudaErrors(cublasCreate(&cublasHandler));
    
    multiply_Den_ClmM_mtxT_mtx(cublasHandler, mtxA_d, mtxB_d, mtxC_d, numOfRowA, numOfColA, numOfColB);

    if(debug){
        printf("\n\n = = = After transpose operation = = =\n");
        printf("\n\n~~mtxA_d~~\n\n");
        print_mtx_clm_d(mtxA_d, numOfRowA, numOfColA);
        printf("\n\n~~mtxB_d~~\n\n");
        print_mtx_clm_d(mtxB_d, numOfRowB, numOfColB);
        printf("\n\n~~mtxC_d~~\n\n");
        print_mtx_clm_d(mtxC_d, numOfRowC, numOfColC);
    }

    checkCudaErrors(cublasDestroy(cublasHandler));
    CHECK(cudaFree(mtxA_d));
    CHECK(cudaFree(mtxB_d));
    CHECK(cudaFree(mtxC_d));
} // end of tranposeTest_Case1




void transposeDenseMultiplyTest_Case2()
{
    const int M = 5;
    const int K = 3;
    const int N = 2;
    
    double mtxA[] = {4.0, 1.0, 1.0, 2.0, 3.0, 
                    1.0, 3.0, 1.0, 5.5, 2.1,
                    1.1, 1.2, 2.3, 1.4, 1.5};

    double mtxB[] = {1.0, 2.0, 3.0, 2.1, 4.4, 
                    4.0, 5.0, 6.0, 1.0, 1.2};

    double* mtxA_d = NULL;
    double* mtxB_d = NULL;
    double* mtxC_d = NULL;


    int numOfRowA = M;
    int numOfColA = K;
    int numOfRowB = M;
    int numOfColB = N;
    int numOfRowC = K;
    int numOfColC = N;

    bool debug = true;

    //Crete handler
    cusolverDnHandle_t cusolverHandler;
    checkCudaErrors(cusolverDnCreate(&cusolverHandler));


    //(1) Allocate device memeory
    CHECK(cudaMalloc((void**)&mtxA_d, numOfRowA * numOfColA * sizeof(double)));
    CHECK(cudaMalloc((void**)&mtxB_d, numOfRowB * numOfColB * sizeof(double)));
    CHECK(cudaMalloc((void**)&mtxC_d, numOfRowC * numOfColC * sizeof(double)));

    //(2) Copy data to device
    CHECK(cudaMemcpy(mtxA_d, mtxA, numOfRowA * numOfColA * sizeof(double), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(mtxB_d, mtxB, numOfRowB * numOfColB * sizeof(double), cudaMemcpyHostToDevice));

    if(debug){
        printf("\n\n = = = Before transpose operation = = =\n");
        printf("\n\n~~mtxA_d~~\n\n");
        print_mtx_clm_d(mtxA_d, numOfRowA, numOfColA);
        printf("\n\n~~mtxB_d~~\n\n");
        print_mtx_clm_d(mtxB_d, numOfRowB, numOfColB);
    }


    //(4) mtxC <- mtxA' * mtxB
    cublasHandle_t cublasHandler = NULL;
    checkCudaErrors(cublasCreate(&cublasHandler));
    
    multiply_Den_ClmM_mtxT_mtx(cublasHandler, mtxA_d, mtxB_d, mtxC_d, numOfRowA, numOfColA, numOfColB);

    if(debug){
        printf("\n\n = = = After transpose operation = = =\n");
        printf("\n\n~~mtxA_d~~\n\n");
        print_mtx_clm_d(mtxA_d, numOfRowA, numOfColA);
        printf("\n\n~~mtxB_d~~\n\n");
        print_mtx_clm_d(mtxB_d, numOfRowB, numOfColB);
        printf("\n\n~~mtxC_d~~\n\n");
        print_mtx_clm_d(mtxC_d, numOfRowC, numOfColC);
    }

    checkCudaErrors(cublasDestroy(cublasHandler));
    CHECK(cudaFree(mtxA_d));
    CHECK(cudaFree(mtxB_d));
    CHECK(cudaFree(mtxC_d));

} // end of tranposeTest_Case2




void transposeDenseMultiplyTest_Case3()
{
    const int M = 7;
    const int K = 5;
    const int N = 4;
    
    double mtxA[] = {4.0, 1.0, 1.0, 2.0, 3.0, -1.1, -1.3, 
                    1.0, 3.0, 1.0, 5.5, 2.1, -2.1, -2.2,
                    1.1, 1.2, 2.3, 1.4, 1.5, 3.5, 4.7,
                    0.9, 0.8, 0.6, 0.8, 0.3, 0.7, 1.1,
                    0.5, 0.7, 2.1, 1.3, -2.1, 1.5, -1.0,};

    double mtxB[] = {1.0, 2.0, 3.0, 2.1, 4.4, 3.2, 2.1, 
                    4.0, 5.0, 6.0, 1.0, 1.2, 5.3, 0.7,
                    3.9, 3.7, 2.4, 2.1, 1.9, 4.8, 1.5,
                    2.3, -0.9, 1.5, 2.7, -1.9, 2.3, 0.7,};

    double* mtxA_d = NULL;
    double* mtxB_d = NULL;
    double* mtxC_d = NULL;


    int numOfRowA = M;
    int numOfColA = K;
    int numOfRowB = M;
    int numOfColB = N;
    int numOfRowC = K;
    int numOfColC = N;

    bool debug = true;

    //Crete handler
    cusolverDnHandle_t cusolverHandler;
    checkCudaErrors(cusolverDnCreate(&cusolverHandler));


    //(1) Allocate device memeory
    CHECK(cudaMalloc((void**)&mtxA_d, numOfRowA * numOfColA * sizeof(double)));
    CHECK(cudaMalloc((void**)&mtxB_d, numOfRowB * numOfColB * sizeof(double)));
    CHECK(cudaMalloc((void**)&mtxC_d, numOfRowC * numOfColC * sizeof(double)));

    //(2) Copy data to device
    CHECK(cudaMemcpy(mtxA_d, mtxA, numOfRowA * numOfColA * sizeof(double), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(mtxB_d, mtxB, numOfRowB * numOfColB * sizeof(double), cudaMemcpyHostToDevice));

    if(debug){
        printf("\n\n = = = Before transpose operation = = =\n");
        printf("\n\n~~mtxA_d~~\n\n");
        print_mtx_clm_d(mtxA_d, numOfRowA, numOfColA);
        printf("\n\n~~mtxB_d~~\n\n");
        print_mtx_clm_d(mtxB_d, numOfRowB, numOfColB);
    }


    //(4) mtxC <- mtxA' * mtxB
    cublasHandle_t cublasHandler = NULL;
    checkCudaErrors(cublasCreate(&cublasHandler));
    
    multiply_Den_ClmM_mtxT_mtx(cublasHandler, mtxA_d, mtxB_d, mtxC_d, numOfRowA, numOfColA, numOfColB);

    if(debug){
        printf("\n\n = = = After transpose operation = = =\n");
        printf("\n\n~~mtxA_d~~\n\n");
        print_mtx_clm_d(mtxA_d, numOfRowA, numOfColA);
        printf("\n\n~~mtxB_d~~\n\n");
        print_mtx_clm_d(mtxB_d, numOfRowB, numOfColB);
        printf("\n\n~~mtxC_d~~\n\n");
        print_mtx_clm_d(mtxC_d, numOfRowC, numOfColC);
    }

    checkCudaErrors(cublasDestroy(cublasHandler));
    CHECK(cudaFree(mtxA_d));
    CHECK(cudaFree(mtxB_d));
    CHECK(cudaFree(mtxC_d));

} // end of tranposeTest_Case3




void tranposeDenseMultiplyTest_Case4()
{

} // end of tranposeTest_Case4




void transposeDenseMultiplyTest_Case5()
{

} // end of tranposeTest_Case5