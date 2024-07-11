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


void computeConditionNumberTest_Case1();
void computeConditionNumberTest_Case2();
void computeConditionNumberTest_Case3();
void computeConditionNumberTest_Case4();
void computeConditionNumberTest_Case5();
void computeConditionNumberTest_Case6();
void computeConditionNumberTest_Case7();
void computeConditionNumberTest_Case8();


int main(int arg, char** argv)
{
    printf("\n\n= = = =computeConditionNumberTest.cu= = = = \n\n");

    // printf("\n\n🔍🔍🔍 Test Case 1 🔍🔍🔍\n\n");
    // computeConditionNumberTest_Case1();

    // printf("\n\n🔍🔍🔍 Test Case 2 🔍🔍🔍\n\n");
    // computeConditionNumberTest_Case2();

    // printf("\n\n🔍🔍🔍 Test Case 3 🔍🔍🔍\n\n");
    // computeConditionNumberTest_Case3();

    // printf("\n\n🔍🔍🔍 Test Case 4 🔍🔍🔍\n\n");
    // computeConditionNumberTest_Case4();

    // printf("\n\n🔍🔍🔍 Test Case 5 🔍🔍🔍\n\n");
    // computeConditionNumberTest_Case5();

    // printf("\n\n🔍🔍🔍 Test Case 6 🔍🔍🔍\n\n");
    // computeConditionNumberTest_Case6();

    // printf("\n\n🔍🔍🔍 Test Case 7 🔍🔍🔍\n\n");
    // computeConditionNumberTest_Case7();
    
    printf("\n\n🔍🔍🔍 Test Case 8 🔍🔍🔍\n\n");
    computeConditionNumberTest_Case8();

    printf("\n\n= = = = end of computeConditionNumberTest = = = =\n\n");
    
    return 0;
} // end of main


void computeConditionNumberTest_Case1()
{
    float mtxA[] = {4.0, 1.0, 1.0, 3.0};
    int numOfRow = 2;
    int numOfClm = 2;


    float* mtxA_d = NULL;

    //(1) Allocate memeory in device
    CHECK(cudaMalloc((void**)&mtxA_d, numOfRow * numOfClm * sizeof(float)));

    //(2) Copy value to device
    CHECK(cudaMemcpy(mtxA_d, mtxA, numOfRow * numOfClm * sizeof(float), cudaMemcpyHostToDevice));
    
    //Calculate condition number with SVD Decomp
    float conditionNum = computeConditionNumber(mtxA_d, numOfRow, numOfClm);
    printf("\n\nCondition Number: %f", conditionNum);
    
    CHECK(cudaFree(mtxA_d));
} // end of case 1

void computeConditionNumberTest_Case2()
{

    float mtxA[] = {4.0, 1.0, 1.0, 
                    1.0, 3.0, 1.0,
                    1.0, 1.0, 5.0};
    int numOfRow = 3;
    int numOfClm = 3;

    float* mtxA_d = NULL;

    //(1) Allocate memeory in device
    CHECK(cudaMalloc((void**)&mtxA_d, numOfRow * numOfClm * sizeof(float)));

    //(2) Copy value to device
    CHECK(cudaMemcpy(mtxA_d, mtxA, numOfRow * numOfClm * sizeof(float), cudaMemcpyHostToDevice));
    
    //Calculate condition number with SVD Decomp
    float conditionNum = computeConditionNumber(mtxA_d, numOfRow, numOfClm);
    printf("\n\nCondition Number: %f", conditionNum);

    CHECK(cudaFree(mtxA_d));
} // end of case 2

void computeConditionNumberTest_Case3()
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
    int numOfRow = 16;
    int numOfClm = 16;

    float* mtxA_d = NULL;


    //(1) Allocate memeory in device
    CHECK(cudaMalloc((void**)&mtxA_d, numOfRow * numOfClm * sizeof(float)));

    //(2) Copy value to device
    CHECK(cudaMemcpy(mtxA_d, mtxA, numOfRow * numOfClm * sizeof(float), cudaMemcpyHostToDevice));
    
    //Calculate condition number with SVD Decomp
    float conditionNum = computeConditionNumber(mtxA_d, numOfRow, numOfClm);
    printf("\n\nCondition Number: %f", conditionNum);

    CHECK(cudaFree(mtxA_d));
} // end of case 3

void computeConditionNumberTest_Case4()
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

    int numOfRow = 19;
    int numOfClm = 19;
    
    float* mtxA_d = NULL;

    //(1) Allocate memeory in device
    CHECK(cudaMalloc((void**)&mtxA_d, numOfRow * numOfClm * sizeof(float)));

    //(2) Copy value to device
    CHECK(cudaMemcpy(mtxA_d, mtxA, numOfRow * numOfClm * sizeof(float), cudaMemcpyHostToDevice));
    
    //Calculate condition number with SVD Decomp
    float conditionNum = computeConditionNumber(mtxA_d, numOfRow, numOfClm);
    printf("\n\nCondition Number: %f", conditionNum);

    CHECK(cudaFree(mtxA_d));
} // end of case 4

void computeConditionNumberTest_Case5()
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

    int numOfRow = 19;
    int numOfClm = 19;
    float* mtxA_d = NULL;

    //(1) Allocate memeory in device
    CHECK(cudaMalloc((void**)&mtxA_d, numOfRow * numOfClm * sizeof(float)));

    //(2) Copy value to device
    CHECK(cudaMemcpy(mtxA_d, mtxA, numOfRow * numOfClm * sizeof(float), cudaMemcpyHostToDevice));
    
    //Calculate condition number with SVD Decomp
    float conditionNum = computeConditionNumber(mtxA_d, numOfRow, numOfClm);
    printf("\n\nCondition Number: %f", conditionNum);
    
    CHECK(cudaFree(mtxA_d));
} // end of case 5

void computeConditionNumberTest_Case6()
{
    
    const int N = 10000;
    float* mtxA_h = generateWellConditoinedSPDMatrix(N);
    float* mtxA_d = NULL;

    //(1) Allocate memeory in device
    CHECK(cudaMalloc((void**)&mtxA_d, N * N * sizeof(float)));

    //(2) Copy value to device
    CHECK(cudaMemcpy(mtxA_d, mtxA_h, N * N * sizeof(float), cudaMemcpyHostToDevice));
    
    //Calculate condition number with SVD Decomp
    float conditionNum = computeConditionNumber(mtxA_d, N, N);
    printf("\n\nCondition Number: %f", conditionNum);

    CHECK(cudaFree(mtxA_d));
    free(mtxA_h);
}

void computeConditionNumberTest_Case7()
{
    
    const int N = 4;
    bool debug = true;

    CSRMatrix csrMtx = generateSparseSPDMatrixCSR(N);

    //print CSR matrix
    if(debug){
        printf("\n\n~~mtxA Sparse CSR format ~~ \n");
        print_CSRMtx(csrMtx);
    }
    

    //Converst CSR to dense matrix
    float* dnsMtx = csrToDense(csrMtx);

    float *mtxA_d = NULL;
    CHECK(cudaMalloc((void**)&mtxA_d, N * N * sizeof(float)));
    CHECK(cudaMemcpy(mtxA_d, dnsMtx, N * N * sizeof(float), cudaMemcpyHostToDevice));

    if(debug){
        printf("\n\n~~mtxA~~\n\n");
        print_mtx_row_d(mtxA_d, N, N);
    }
    
    float conditionNum = computeConditionNumber(mtxA_d, N, N);
    printf("\n\nCondition Number: %f\n\n", conditionNum);

    //Free memory
    CHECK(cudaFree(mtxA_d));
    free(dnsMtx);
    freeCSRMatrix(csrMtx);
}

void computeConditionNumberTest_Case8()
{
    
    const int N = 10000;
    bool debug = false;

    CSRMatrix csrMtx = generateSparseSPDMatrixCSR(N);

    //print CSR matrix
    if(debug){
        printf("\n\n~~mtxA Sparse CSR format ~~ \n");
        print_CSRMtx(csrMtx);
    }
    

    //Converst CSR to dense matrix
    float* dnsMtx = csrToDense(csrMtx);

    float *mtxA_d = NULL;
    CHECK(cudaMalloc((void**)&mtxA_d, N * N * sizeof(float)));
    CHECK(cudaMemcpy(mtxA_d, dnsMtx, N * N * sizeof(float), cudaMemcpyHostToDevice));

    if(debug){
        printf("\n\n~~mtxA~~\n\n");
        print_mtx_row_d(mtxA_d, N, N);
    }
    
    float conditionNum = computeConditionNumber(mtxA_d, N, N);
    printf("\n\nCondition Number: %f\n\n", conditionNum);

    //Free memory
    CHECK(cudaFree(mtxA_d));
    free(dnsMtx);
    freeCSRMatrix(csrMtx);
}