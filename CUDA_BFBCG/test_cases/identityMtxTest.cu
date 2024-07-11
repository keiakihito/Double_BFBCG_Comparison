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

void identityMtxTest_Case1();
void identityMtxTest_Case2();
void identityMtxTest_Case3();
void identityMtxTest_Case4();
void identityMtxTest_Case5();

int main(int arg, char** argv)
{
    printf("\n\n= = = identityMtx Test= = = \n\n");

    // printf("\n\n🔍🔍🔍Test Case 1🔍🔍🔍\n\n");
    // identityMtxTest_Case1();

    // printf("\n\n🔍🔍🔍Test Case 2🔍🔍🔍\n\n");
    // identityMtxTest_Case2();

    // printf("\n\n🔍🔍🔍Test Case 3🔍🔍🔍\n\n");
    // identityMtxTest_Case3();

    // printf("\n\n🔍🔍🔍Test Case 4🔍🔍🔍\n\n");
    // identityMtxTest_Case4();

    printf("\n\n🔍🔍🔍Test Case 5🔍🔍🔍\n\n");
    identityMtxTest_Case5();

    printf("\n\n= = = End of identityMtx Test = = = \n\n");
    return 0;
}

void identityMtxTest_Case1()
{   
    int N = 2;
    float* mtxI_d = NULL;

    //Allocate memory in device
    CHECK(cudaMalloc((void**)&mtxI_d, N * N * sizeof(float)));

    //Make Identity matrix
    createIdentityMtx(mtxI_d, N);

    print_mtx_clm_d(mtxI_d, N, N);

    //Free memeory
    CHECK(cudaFree(mtxI_d));
    
} // end of identityMtxTest_Case1

void identityMtxTest_Case2()
{
    int N = 8;
    float* mtxI_d = NULL;

    //Allocate memory in device
    CHECK(cudaMalloc((void**)&mtxI_d, N * N * sizeof(float)));

    //Make Identity matrix
    createIdentityMtx(mtxI_d, N);

    print_mtx_clm_d(mtxI_d, N, N);

    //Free memeory
    CHECK(cudaFree(mtxI_d));

}// end of identityMtxTest_Case2

void identityMtxTest_Case3()
{
    int N = 17;
    float* mtxI_d = NULL;

    //Allocate memory in device
    CHECK(cudaMalloc((void**)&mtxI_d, N * N * sizeof(float)));

    //Make Identity matrix
    createIdentityMtx(mtxI_d, N);

    print_mtx_clm_d(mtxI_d, N, N);

    //Free memeory
    CHECK(cudaFree(mtxI_d));

}// end of identityMtxTest_Case3

void identityMtxTest_Case4()
{
    int N = 65;
    float* mtxI_d = NULL;

    //Allocate memory in device
    CHECK(cudaMalloc((void**)&mtxI_d, N * N * sizeof(float)));

    //Make Identity matrix
    createIdentityMtx(mtxI_d, N);

    print_mtx_clm_d(mtxI_d, N, N);

    //Free memeory
    CHECK(cudaFree(mtxI_d));

}// end of identityMtxTest_Case4

void identityMtxTest_Case5()
{
    int N = 2049;
    float* mtxI_d = NULL;

    //Allocate memory in device
    CHECK(cudaMalloc((void**)&mtxI_d, N * N * sizeof(float)));

    //Make Identity matrix
    createIdentityMtx(mtxI_d, N);

    print_mtx_clm_d(mtxI_d, N, N);

    //Free memeory
    CHECK(cudaFree(mtxI_d));
    
}// end of identityMtxTest_Case5