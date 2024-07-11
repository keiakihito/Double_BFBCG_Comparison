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


#define CHECK(call){ \
    const cudaError_t cuda_ret = call; \
    if(cuda_ret != cudaSuccess){ \
        printf("Error: %s:%d,  ", __FILE__, __LINE__ );\
        printf("code: %d, reason: %s \n", cuda_ret, cudaGetErrorString(cuda_ret));\
        exit(-1); \
    }\
}


void inverseTest_Case1();
void inverseTest_Case2();
void inverseTest_Case3();
void inverseTest_Case4();
void inverseTest_Case5();

int main(int arg, char** argv)
{
    
    printf("\n\n= = = =inverseTest.cu= = = = \n\n");
    
    // printf("\n\n🔍🔍🔍 Test Case 1 🔍🔍🔍\n\n");
    // inverseTest_Case1();

    // printf("\n\n🔍🔍🔍 Test Case 2 🔍🔍🔍\n\n");
    // inverseTest_Case2();

    // printf("\n\n🔍🔍🔍 Test Case 3 🔍🔍🔍\n\n");
    // inverseTest_Case3();

    // printf("\n\n🔍🔍🔍 Test Case 4 🔍🔍🔍\n\n");
    // inverseTest_Case4();

    printf("\n\n🔍🔍🔍 Test Case 5 🔍🔍🔍\n\n");
    inverseTest_Case5();

    printf("\n\n= = = = end of invereTest = = = =\n\n");

    return 0;
} // end of main


void inverseTest_Case1()
{

    /*
    Compute inverse, X with LU factorization
    A = LU
    LU * X = I
    L *(UX) = L * Y = I

    Solve X
        UX = Y

    */

    /*
    mtxA =  |4 1|
            |1 3|

    Answer
    mtxA^(-1) = | 3/11 -1/11|
                |-1/11  4/11|

    or

    mtxA^(-1) = |0.2727  -0.091|
                |-0.091  0.3636|
    */

    //Defince the dense matrix A column major
    float mtxA[] = {4.0, 1.0, 1.0, 3.0};
    float* mtxA_d = NULL;
    float* mtxA_inv_d = NULL;

    const int N = 2;
    

    bool debug = true;

    //Crete handler
    cusolverDnHandle_t cusolverHandler;
    checkCudaErrors(cusolverDnCreate(&cusolverHandler));


    //(1) Allocate device memeory
    CHECK(cudaMalloc((void**)&mtxA_d, N * N * sizeof(float)));
    CHECK(cudaMalloc((void**)&mtxA_inv_d, N * N * sizeof(float)));

    //(2) Copy data to device
    CHECK(cudaMemcpy(mtxA_d, mtxA, N * N * sizeof(float), cudaMemcpyHostToDevice));
    
    if(debug){
        printf("\n\n~~mtxA_d~~\n\n");
        print_mtx_clm_d(mtxA_d, N, N);
    }


    //(3) Perform inverse operation
    inverse_Den_Mtx(cusolverHandler, mtxA_d, mtxA_inv_d, N);

    //(4) Check the result
    if(debug){
        printf("\n\nCompute inverse\n");
        printf("\n\n~~mtxA inverse~~\n\n");
        print_mtx_clm_d(mtxA_inv_d, N, N);
    }

    //(5) Free memory
    checkCudaErrors(cusolverDnDestroy(cusolverHandler));
    CHECK(cudaFree(mtxA_d));
    CHECK(cudaFree(mtxA_inv_d));

} // end of inverseTest_Case1


void inverseTest_Case2()
{
    float mtxA[] = {4.0, 1.0, 1.0, 
                    1.0, 3.0, 1.0,
                    1.0, 1.0, 5.0};
    float* mtxA_d = NULL;
    float* mtxA_inv_d = NULL;

    const int N = 3;

    bool debug = true;

    //Crete handler
    cusolverDnHandle_t cusolverHandler;
    checkCudaErrors(cusolverDnCreate(&cusolverHandler));


    //(1) Allocate device memeory
    CHECK(cudaMalloc((void**)&mtxA_d, N * N * sizeof(float)));
    CHECK(cudaMalloc((void**)&mtxA_inv_d, N * N * sizeof(float)));

    //(2) Copy data to device
    CHECK(cudaMemcpy(mtxA_d, mtxA, N * N * sizeof(float), cudaMemcpyHostToDevice));
    
    if(debug){
        printf("\n\n~~mtxA_d~~\n\n");
        print_mtx_clm_d(mtxA_d, N, N);
    }


    //(3) Perform inverse operation
    inverse_Den_Mtx(cusolverHandler, mtxA_d, mtxA_inv_d, N);

    //(4) Check the result
    if(debug){
        printf("\n\nCompute inverse\n");
        printf("\n\n~~mtxA inverse~~\n\n");
        print_mtx_clm_d(mtxA_inv_d, N, N);
    }

    //(5) Free memory
    checkCudaErrors(cusolverDnDestroy(cusolverHandler));
    CHECK(cudaFree(mtxA_d));
    CHECK(cudaFree(mtxA_inv_d));

}// end of inverseTest_Case2


void inverseTest_Case3()
{
    float mtxA[] = {
        4.0,  1.0,  0.0,  0.0,  1.0,  0.0,  0.0,  0.0,  1.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,
        1.0,  4.0,  1.0,  0.0,  0.0,  1.0,  0.0,  0.0,  0.0,  1.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,
        0.0,  1.0,  4.0,  1.0,  0.0,  0.0,  1.0,  0.0,  0.0,  0.0,  1.0,  0.0,  0.0,  0.0,  0.0,  0.0,
        0.0,  0.0,  1.0,  4.0,  1.0,  0.0,  0.0,  1.0,  0.0,  0.0,  0.0,  1.0,  0.0,  0.0,  0.0,  0.0,
        1.0,  0.0,  0.0,  1.0,  4.0,  1.0,  0.0,  0.0,  1.0,  0.0,  0.0,  0.0,  1.0,  0.0,  0.0,  0.0,
        0.0,  1.0,  0.0,  0.0,  1.0,  4.0,  1.0,  0.0,  0.0,  1.0,  0.0,  0.0,  0.0,  1.0,  0.0,  0.0,
        0.0,  0.0,  1.0,  0.0,  0.0,  1.0,  4.0,  1.0,  0.0,  0.0,  1.0,  0.0,  0.0,  0.0,  1.0,  0.0,
        0.0,  0.0,  0.0,  1.0,  0.0,  0.0,  1.0,  4.0,  1.0,  0.0,  0.0,  1.0,  0.0,  0.0,  0.0,  1.0,
        1.0,  0.0,  0.0,  0.0,  1.0,  0.0,  0.0,  1.0,  4.0,  1.0,  0.0,  0.0,  1.0,  0.0,  0.0,  0.0,
        0.0,  1.0,  0.0,  0.0,  0.0,  1.0,  0.0,  0.0,  1.0,  4.0,  1.0,  0.0,  0.0,  1.0,  0.0,  0.0,
        0.0,  0.0,  1.0,  0.0,  0.0,  0.0,  1.0,  0.0,  0.0,  1.0,  4.0,  1.0,  0.0,  0.0,  1.0,  0.0,
        0.0,  0.0,  0.0,  1.0,  0.0,  0.0,  0.0,  1.0,  0.0,  0.0,  1.0,  4.0,  1.0,  0.0,  0.0,  1.0,
        0.0,  0.0,  0.0,  0.0,  1.0,  0.0,  0.0,  0.0,  1.0,  0.0,  0.0,  1.0,  4.0,  1.0,  0.0,  0.0,
        0.0,  0.0,  0.0,  0.0,  0.0,  1.0,  0.0,  0.0,  0.0,  1.0,  0.0,  0.0,  1.0,  4.0,  1.0,  0.0,
        0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  1.0,  0.0,  0.0,  0.0,  1.0,  0.0,  0.0,  1.0,  4.0,  1.0,
        0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  1.0,  0.0,  0.0,  0.0,  1.0,  0.0,  0.0,  1.0,  4.0
    };

    float* mtxA_d = NULL;
    float* mtxA_inv_d = NULL;

    const int N = 16;

    bool debug = true;

    //Crete handler
    cusolverDnHandle_t cusolverHandler;
    checkCudaErrors(cusolverDnCreate(&cusolverHandler));


    //(1) Allocate device memeory
    CHECK(cudaMalloc((void**)&mtxA_d, N * N * sizeof(float)));
    CHECK(cudaMalloc((void**)&mtxA_inv_d, N * N * sizeof(float)));

    //(2) Copy data to device
    CHECK(cudaMemcpy(mtxA_d, mtxA, N * N * sizeof(float), cudaMemcpyHostToDevice));
    
    if(debug){
        printf("\n\n~~mtxA_d~~\n\n");
        print_mtx_clm_d(mtxA_d, N, N);
    }


    //(3) Perform inverse operation
    inverse_Den_Mtx(cusolverHandler, mtxA_d, mtxA_inv_d, N);

    //(4) Check the result
    if(debug){
        printf("\n\nCompute inverse\n");
        printf("\n\n~~mtxA inverse~~\n\n");
        print_mtx_clm_d(mtxA_inv_d, N, N);
    }

    //(5) Free memory
    checkCudaErrors(cusolverDnDestroy(cusolverHandler));
    CHECK(cudaFree(mtxA_d));
    CHECK(cudaFree(mtxA_inv_d));
    
}// end of inverseTest_Case3


void inverseTest_Case4()
{
    float mtxA[] = {
        10.123456, 0.234567, 0.345678, 0.456789, 0.567890, 0.678901, 0.789012, 0.890123, 0.901234, 0.012345, 0.123456, 0.234567, 0.345678, 0.456789, 0.567890, 0.678901, 0.789012, 0.890123, 0.901234,
        0.234567, 11.345678, 0.456789, 0.567890, 0.678901, 0.789012, 0.890123, 0.901234, 0.012345, 0.123456, 0.234567, 0.345678, 0.456789, 0.567890, 0.678901, 0.789012, 0.890123, 0.901234, 0.012345,
        0.345678, 0.456789, 12.567890, 0.678901, 0.789012, 0.890123, 0.901234, 0.012345, 0.123456, 0.234567, 0.345678, 0.456789, 0.567890, 0.678901, 0.789012, 0.890123, 0.901234, 0.012345, 0.123456,
        0.456789, 0.567890, 0.678901, 13.789012, 0.890123, 0.901234, 0.012345, 0.123456, 0.234567, 0.345678, 0.456789, 0.567890, 0.678901, 0.789012, 0.890123, 0.901234, 0.012345, 0.123456, 0.234567,
        0.567890, 0.678901, 0.789012, 0.890123, 14.901234, 0.012345, 0.123456, 0.234567, 0.345678, 0.456789, 0.567890, 0.678901, 0.789012, 0.890123, 0.901234, 0.012345, 0.123456, 0.234567, 0.345678,
        0.678901, 0.789012, 0.890123, 0.901234, 0.012345, 15.123456, 0.234567, 0.345678, 0.456789, 0.567890, 0.678901, 0.789012, 0.890123, 0.901234, 0.012345, 0.123456, 0.234567, 0.345678, 0.456789,
        0.789012, 0.890123, 0.901234, 0.012345, 0.123456, 0.234567, 16.345678, 0.456789, 0.567890, 0.678901, 0.789012, 0.890123, 0.901234, 0.012345, 0.123456, 0.234567, 0.345678, 0.456789, 0.567890,
        0.890123, 0.901234, 0.012345, 0.123456, 0.234567, 0.345678, 0.456789, 17.567890, 0.678901, 0.789012, 0.890123, 0.901234, 0.012345, 0.123456, 0.234567, 0.345678, 0.456789, 0.567890, 0.678901,
        0.901234, 0.012345, 0.123456, 0.234567, 0.345678, 0.456789, 0.567890, 0.678901, 18.789012, 0.890123, 0.901234, 0.012345, 0.123456, 0.234567, 0.345678, 0.456789, 0.567890, 0.678901, 0.789012,
        0.012345, 0.123456, 0.234567, 0.345678, 0.456789, 0.567890, 0.678901, 0.789012, 0.890123, 19.901234, 0.012345, 0.123456, 0.234567, 0.345678, 0.456789, 0.567890, 0.678901, 0.789012, 0.890123,
        0.123456, 0.234567, 0.345678, 0.456789, 0.567890, 0.678901, 0.789012, 0.890123, 0.901234, 0.012345, 20.123456, 0.234567, 0.345678, 0.456789, 0.567890, 0.678901, 0.789012, 0.890123, 0.901234,
        0.234567, 0.345678, 0.456789, 0.567890, 0.678901, 0.789012, 0.890123, 0.901234, 0.012345, 0.123456, 0.234567, 19.345678, 0.456789, 0.567890, 0.678901, 0.789012, 0.890123, 0.901234, 0.012345,
        0.345678, 0.456789, 0.567890, 0.678901, 0.789012, 0.890123, 0.901234, 0.012345, 0.123456, 0.234567, 0.345678, 0.456789, 18.567890, 0.678901, 0.789012, 0.890123, 0.901234, 0.012345, 0.123456,
        0.456789, 0.567890, 0.678901, 0.789012, 0.890123, 0.901234, 0.012345, 0.123456, 0.234567, 0.345678, 0.456789, 0.567890, 0.678901, 17.789012, 0.890123, 0.901234, 0.012345, 0.123456, 0.234567,
        0.567890, 0.678901, 0.789012, 0.890123, 0.901234, 0.012345, 0.123456, 0.234567, 0.345678, 0.456789, 0.567890, 0.678901, 0.789012, 0.890123, 16.901234, 0.012345, 0.123456, 0.234567, 0.345678,
        0.678901, 0.789012, 0.890123, 0.901234, 0.012345, 0.123456, 0.234567, 0.345678, 0.456789, 0.567890, 0.678901, 0.789012, 0.890123, 0.901234, 0.012345, 15.123456, 0.234567, 0.345678, 0.456789,
        0.789012, 0.890123, 0.901234, 0.012345, 0.123456, 0.234567, 0.345678, 0.456789, 0.567890, 0.678901, 0.789012, 0.890123, 0.901234, 0.012345, 0.123456, 0.234567, 14.345678, 0.456789, 0.567890,
        0.890123, 0.901234, 0.012345, 0.123456, 0.234567, 0.345678, 0.456789, 0.567890, 0.678901, 0.789012, 0.890123, 0.901234, 0.012345, 0.123456, 0.234567, 0.345678, 0.456789, 13.567890, 0.678901,
        0.901234, 0.012345, 0.123456, 0.234567, 0.345678, 0.456789, 0.567890, 0.678901, 0.789012, 0.890123, 0.901234, 0.012345, 0.123456, 0.234567, 0.345678, 0.456789, 0.567890, 0.678901, 12.789012
    };

    float* mtxA_d = NULL;
    float* mtxA_inv_d = NULL;

    const int N = 19;

    bool debug = true;

    //Crete handler
    cusolverDnHandle_t cusolverHandler;
    checkCudaErrors(cusolverDnCreate(&cusolverHandler));


    //(1) Allocate device memeory
    CHECK(cudaMalloc((void**)&mtxA_d, N * N * sizeof(float)));
    CHECK(cudaMalloc((void**)&mtxA_inv_d, N * N * sizeof(float)));

    //(2) Copy data to device
    CHECK(cudaMemcpy(mtxA_d, mtxA, N * N * sizeof(float), cudaMemcpyHostToDevice));
    
    if(debug){
        printf("\n\n~~mtxA_d~~\n\n");
        print_mtx_clm_d(mtxA_d, N, N);
    }


    //(3) Perform inverse operation
    inverse_Den_Mtx(cusolverHandler, mtxA_d, mtxA_inv_d, N);

    //(4) Check the result
    if(debug){
        printf("\n\nCompute inverse\n");
        printf("\n\n~~mtxA inverse~~\n\n");
        print_mtx_clm_d(mtxA_inv_d, N, N);
    }

    //(5) Free memory
    checkCudaErrors(cusolverDnDestroy(cusolverHandler));
    CHECK(cudaFree(mtxA_d));
    CHECK(cudaFree(mtxA_inv_d));
}// end of inverseTest_Case4


void inverseTest_Case5()
{
    float mtxA[] = {
        0.123456, 0.234567, 0.345678, 0.456789, 0.567890, 0.678901, 0.789012, 0.890123, 0.901234, 0.012345, 0.123456, 0.234567, 0.345678, 0.456789, 0.567890, 0.678901, 0.789012, 0.890123, 0.901234,
        0.234567, 0.345678, 0.456789, 0.567890, 0.678901, 0.789012, 0.890123, 0.901234, 0.012345, 0.123456, 0.234567, 0.345678, 0.456789, 0.567890, 0.678901, 0.789012, 0.890123, 0.901234, 0.012345,
        0.345678, 0.456789, 0.567890, 0.678901, 0.789012, 0.890123, 0.901234, 0.012345, 0.123456, 0.234567, 0.345678, 0.456789, 0.567890, 0.678901, 0.789012, 0.890123, 0.901234, 0.012345, 0.123456,
        0.456789, 0.567890, 0.678901, 0.789012, 0.890123, 0.901234, 0.012345, 0.123456, 0.234567, 0.345678, 0.456789, 0.567890, 0.678901, 0.789012, 0.890123, 0.901234, 0.012345, 0.123456, 0.234567,
        0.567890, 0.678901, 0.789012, 0.890123, 0.901234, 0.012345, 0.123456, 0.234567, 0.345678, 0.456789, 0.567890, 0.678901, 0.789012, 0.890123, 0.901234, 0.012345, 0.123456, 0.234567, 0.345678,
        0.678901, 0.789012, 0.890123, 0.901234, 0.012345, 0.123456, 0.234567, 0.345678, 0.456789, 0.567890, 0.678901, 0.789012, 0.890123, 0.901234, 0.012345, 0.123456, 0.234567, 0.345678, 0.456789,
        0.789012, 0.890123, 0.901234, 0.012345, 0.123456, 0.234567, 0.345678, 0.456789, 0.567890, 0.678901, 0.789012, 0.890123, 0.901234, 0.012345, 0.123456, 0.234567, 0.345678, 0.456789, 0.567890,
        0.890123, 0.901234, 0.012345, 0.123456, 0.234567, 0.345678, 0.456789, 0.567890, 0.678901, 0.789012, 0.890123, 0.901234, 0.012345, 0.123456, 0.234567, 0.345678, 0.456789, 0.567890, 0.678901,
        0.901234, 0.012345, 0.123456, 0.234567, 0.345678, 0.456789, 0.567890, 0.678901, 0.789012, 0.890123, 0.901234, 0.012345, 0.123456, 0.234567, 0.345678, 0.456789, 0.567890, 0.678901, 0.789012,
        0.012345, 0.123456, 0.234567, 0.345678, 0.456789, 0.567890, 0.678901, 0.789012, 0.890123, 0.901234, 0.012345, 0.123456, 0.234567, 0.345678, 0.456789, 0.567890, 0.678901, 0.789012, 0.890123,
        0.123456, 0.234567, 0.345678, 0.456789, 0.567890, 0.678901, 0.789012, 0.890123, 0.901234, 0.012345, 0.123456, 0.234567, 0.345678, 0.456789, 0.567890, 0.678901, 0.789012, 0.890123, 0.901234,
        0.234567, 0.345678, 0.456789, 0.567890, 0.678901, 0.789012, 0.890123, 0.901234, 0.012345, 0.123456, 0.234567, 0.345678, 0.456789, 0.567890, 0.678901, 0.789012, 0.890123, 0.901234, 0.012345,
        0.345678, 0.456789, 0.567890, 0.678901, 0.789012, 0.890123, 0.901234, 0.012345, 0.123456, 0.234567, 0.345678, 0.456789, 0.567890, 0.678901, 0.789012, 0.890123, 0.901234, 0.012345, 0.123456,
        0.456789, 0.567890, 0.678901, 0.789012, 0.890123, 0.901234, 0.012345, 0.123456, 0.234567, 0.345678, 0.456789, 0.567890, 0.678901, 0.789012, 0.890123, 0.901234, 0.012345, 0.123456, 0.234567,
        0.567890, 0.678901, 0.789012, 0.890123, 0.901234, 0.012345, 0.123456, 0.234567, 0.345678, 0.456789, 0.567890, 0.678901, 0.789012, 0.890123, 0.901234, 0.012345, 0.123456, 0.234567, 0.345678,
        0.678901, 0.789012, 0.890123, 0.901234, 0.012345, 0.123456, 0.234567, 0.345678, 0.456789, 0.567890, 0.678901, 0.789012, 0.890123, 0.901234, 0.012345, 0.123456, 0.234567, 0.345678, 0.456789,
        0.789012, 0.890123, 0.901234, 0.012345, 0.123456, 0.234567, 0.345678, 0.456789, 0.567890, 0.678901, 0.789012, 0.890123, 0.901234, 0.012345, 0.123456, 0.234567, 0.345678, 0.456789, 0.567890,
        0.890123, 0.901234, 0.012345, 0.123456, 0.234567, 0.345678, 0.456789, 0.567890, 0.678901, 0.789012, 0.890123, 0.901234, 0.012345, 0.123456, 0.234567, 0.345678, 0.456789, 0.567890, 0.678901,
        0.901234, 0.012345, 0.123456, 0.234567, 0.345678, 0.456789, 0.567890, 0.678901, 0.789012, 0.890123, 0.901234, 0.012345, 0.123456, 0.234567, 0.345678, 0.456789, 0.567890, 0.678901, 0.789012
    };

    float* mtxA_d = NULL;
    float* mtxA_inv_d = NULL;

    const int N = 19;

    bool debug = true;

    //Crete handler
    cusolverDnHandle_t cusolverHandler;
    checkCudaErrors(cusolverDnCreate(&cusolverHandler));


    //(1) Allocate device memeory
    CHECK(cudaMalloc((void**)&mtxA_d, N * N * sizeof(float)));
    CHECK(cudaMalloc((void**)&mtxA_inv_d, N * N * sizeof(float)));

    //(2) Copy data to device
    CHECK(cudaMemcpy(mtxA_d, mtxA, N * N * sizeof(float), cudaMemcpyHostToDevice));
    
    if(debug){
        printf("\n\n~~mtxA_d~~\n\n");
        print_mtx_clm_d(mtxA_d, N, N);
    }


    //(3) Perform inverse operation
    inverse_Den_Mtx(cusolverHandler, mtxA_d, mtxA_inv_d, N);

    //(4) Check the result
    if(debug){
        printf("\n\nCompute inverse\n");
        printf("\n\n~~mtxA inverse~~\n\n");
        print_mtx_clm_d(mtxA_inv_d, N, N);
    }

    //(5) Free memory
    checkCudaErrors(cusolverDnDestroy(cusolverHandler));
    CHECK(cudaFree(mtxA_d));
    CHECK(cudaFree(mtxA_inv_d));
    
}// end of inverseTest_Case5