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


void stopTest_Case1();
void stopTest_Case2();
void stopTest_Case3();
void stopTest_Case4();
void stopTest_Case5();

int main(int arg, char** argv)
{

    // stopTest_Case1();
    // stopTest_Case2();
    // stopTest_Case3();
    stopTest_Case4();
    stopTest_Case5();

    return 0;
} // end of main


void stopTest_Case1()
{
    float mtxA[] = {
        0.00000001, 0.00000001, 0.00000001,
        0.000000001, 0.00000001, 0.00000001 
    };

    int numOfRow = 3;
    int numOfClm = 2;
    float const THRESHOLD = 1e-6;

    //Create handler
    cublasHandle_t cublasHandler = NULL;
    checkCudaErrors(cublasCreate(&cublasHandler));


    //(1) Allocate memory
    float* mtxA_d = NULL;
    CHECK(cudaMalloc((void**)& mtxA_d, numOfRow * numOfClm * sizeof(float)));
    
    //(2) Copy value from host to device
    CHECK(cudaMemcpy(mtxA_d, mtxA, numOfRow * numOfClm * sizeof(float), cudaMemcpyHostToDevice));
    
    //(3) Check stops or not
    bool isStop = checkStop(cublasHandler, mtxA_d, numOfRow, numOfClm, THRESHOLD);
    printf("\nisStop : %s\n", isStop? "true" : "false");

    //(4) Free
    CHECK(cudaFree(mtxA_d));

}


void stopTest_Case2()
{
    float mtxA[] = {
        0.000001, -0.000001, 0.00000001, 0.0001, 0.0000001, -0.000001, 
        0.000000001, 0.00000001, 0.00000001, 1, 2, 3
    };

    int numOfRow = 6;
    int numOfClm = 2;
    float const THRESHOLD = 1e-6;

    //Create handler
    cublasHandle_t cublasHandler = NULL;
    checkCudaErrors(cublasCreate(&cublasHandler));


    //(1) Allocate memory
    float* mtxA_d = NULL;
    CHECK(cudaMalloc((void**)& mtxA_d, numOfRow * numOfClm * sizeof(float)));
    
    //(2) Copy value from host to device
    CHECK(cudaMemcpy(mtxA_d, mtxA, numOfRow * numOfClm * sizeof(float), cudaMemcpyHostToDevice));
    
    //(3) Check stops or not
    bool isStop = checkStop(cublasHandler, mtxA_d, numOfRow, numOfClm, THRESHOLD);
    printf("\nisStop : %s\n", isStop? "true" : "false");

    //(4) Free
    CHECK(cudaFree(mtxA_d));
}


void stopTest_Case3()
{
    float mtxA[] = {
        0.000000001, 0.00000001, 0.00000001, 1, 0.0000001, -0.000000001, 0.0, 0.1, 0.1, 5,
        0.000001, -0.000001, 0.00000001, 0.0001, 0.0000001, -0.000001, 2, 3, 4, 5,
        0.000001, -0.000001, 0.00000001, 0.0001, 0.0000001, -0.000001, 2, 3, 4, 5 
    };

    int numOfRow = 10;
    int numOfClm = 3;
    float const THRESHOLD = 1e-5;

    //Create handler
    cublasHandle_t cublasHandler = NULL;
    checkCudaErrors(cublasCreate(&cublasHandler));


    //(1) Allocate memory
    float* mtxA_d = NULL;
    CHECK(cudaMalloc((void**)& mtxA_d, numOfRow * numOfClm * sizeof(float)));
    
    //(2) Copy value from host to device
    CHECK(cudaMemcpy(mtxA_d, mtxA, numOfRow * numOfClm * sizeof(float), cudaMemcpyHostToDevice));
    
    //(3) Check stops or not
    bool isStop = checkStop(cublasHandler, mtxA_d, numOfRow, numOfClm, THRESHOLD);
    printf("\nisStop : %s\n", isStop? "true" : "false");

    //(4) Free
    CHECK(cudaFree(mtxA_d)); 
}


void stopTest_Case4()
{
    float mtxA[] = {
        -0.000000001, -0.00000001, 0.00000001, -0.0000001, -0.0000001, -0.000000001, 0.0, 0.000001, 0.000001, -0.00000005,
        0.000001, -0.000001, 0.00000001, 0.0001, 0.0000001, -0.000001, 2, 3, 4, 5,
        0.000001, -0.000001, 0.00000001, 0.0001, 0.0000001, -0.000001, 2, 3, 4, 5 
    };

    int numOfRow = 10;
    int numOfClm = 3;
    float const THRESHOLD = 1e-6;

    //Create handler
    cublasHandle_t cublasHandler = NULL;
    checkCudaErrors(cublasCreate(&cublasHandler));


    //(1) Allocate memory
    float* mtxA_d = NULL;
    CHECK(cudaMalloc((void**)& mtxA_d, numOfRow * numOfClm * sizeof(float)));
    
    //(2) Copy value from host to device
    CHECK(cudaMemcpy(mtxA_d, mtxA, numOfRow * numOfClm * sizeof(float), cudaMemcpyHostToDevice));
    
    //(3) Check stops or not
    bool isStop = checkStop(cublasHandler, mtxA_d, numOfRow, numOfClm, THRESHOLD);
    printf("\nisStop : %s\n", isStop? "true" : "false");

    //(4) Free
    CHECK(cudaFree(mtxA_d)); 
    
}


void stopTest_Case5()
{
    float mtxA[] = {
        -0.000000001, -0.00000001, 0.00000001, -0.0000001, -0.0000001, -0.000000001, 0.0, 0.0000001, 0.0000001, -0.00000005,
        0.000001, -0.000001, 0.00000001, 0.0001, 0.0000001, -0.000001, 2, 3, 4, 5,
        0.000001, -0.000001, 0.00000001, 0.0001, 0.0000001, -0.000001, 2, 3, 4, 5 
    };

    int numOfRow = 10;
    int numOfClm = 3;
    float const THRESHOLD = 1e-6;

    //Create handler
    cublasHandle_t cublasHandler = NULL;
    checkCudaErrors(cublasCreate(&cublasHandler));


    //(1) Allocate memory
    float* mtxA_d = NULL;
    CHECK(cudaMalloc((void**)& mtxA_d, numOfRow * numOfClm * sizeof(float)));
    
    //(2) Copy value from host to device
    CHECK(cudaMemcpy(mtxA_d, mtxA, numOfRow * numOfClm * sizeof(float), cudaMemcpyHostToDevice));
    
    //(3) Check stops or not
    bool isStop = checkStop(cublasHandler, mtxA_d, numOfRow, numOfClm, THRESHOLD);
    printf("\nisStop : %s\n", isStop? "true" : "false");

    //(4) Free
    CHECK(cudaFree(mtxA_d)); 
    
}