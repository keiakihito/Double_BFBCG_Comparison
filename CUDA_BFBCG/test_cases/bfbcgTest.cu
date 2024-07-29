/*
Personal Note to compile this program in NCSA delta.

1. Srun
srun --account=bchn-delta-gpu --partition=gpuA40x4-interactive --nodes=1 --gpus-per-node=1 --tasks-per-node=16 --cpus-per-task=1 --mem=20G --pty bash

2. Comile with this long long command
nvcc bfbcgTest.cu -o bfbcgTest -I/sw/spack/deltas11-2023-03/apps/linux-rhel8-zen3/gcc-11.4.0/cuda-11.8.0-vfixfmc/include -L/sw/spack/deltas11-2023-03/apps/linux-rhel8-zen3/gcc-11.4.0/cuda-11.8.0-vfixfmc/lib64 -lcudart -lcublas -lcusolver -lcusparse -I/sw/spack/deltas11-2023-03/apps/linux-rhel8-zen3/gcc-11.4.0/magma-2.8.0-mvkrj4y/include -L/sw/spack/deltas11-2023-03/apps/linux-rhel8-zen3/gcc-11.4.0/magma-2.8.0-mvkrj4y/lib -L/sw/spack/deltas11-2023-03/apps/linux-rhel8-zen3/gcc-11.4.0/openblas-0.3.25-5yvxjnl/lib -lmagma -lopenblas

3. Set path for magma
 export LD_LIBRARY_PATH=/sw/spack/deltas11-2023-03/apps/linux-rhel8-zen3/gcc-11.4.0/magma-2.8.0-mvkrj4y/lib:/sw/spack/deltas11-2023-03/apps/linux-rhel8-zen3/gcc-11.4.0/cuda-11.8.0-vfixfmc/lib64:$LD_LIBRARY_PATH

 4. 
 ./bfbcgTest
 */




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
#include "../include/utils/checks.h"
#include "../include/functions/helper.h"
#include "../include/functions/cuBLAS_util.h"
#include "../include/functions/cuSPARSE_util.h"
#include "../include/functions/cuSOLVER_util.h"
#include "../include/functions/bfbcg.h"
#include "../include/CSRMatrix.h"







void bfbcgTest_Case1();
void bfbcgTest_Case2();
void bfbcgTest_Case3();
void bfbcgTest_Case4();
void bfbcgTest_Case5();

int main(int arg, char** argv)
{
    
    printf("\n\n= = = =bfbcgTest.cu= = = = \n\n");
    
    // printf("\n\n🔍🔍🔍 Test Case 1 🔍🔍🔍\n\n");
    // bfbcgTest_Case1();

    // printf("\n\n🔍🔍🔍 Test Case 2 🔍🔍🔍\n\n");
    // bfbcgTest_Case2();

    // printf("\n\n🔍🔍🔍 Test Case 3 🔍🔍🔍\n\n");
    // bfbcgTest_Case3();

    printf("\n\n🔍🔍🔍 Test Case 4 🔍🔍🔍\n\n");
    bfbcgTest_Case4();


    printf("\n\n= = = = end of bfbcgTest = = = =\n\n");

    return 0;
} // end of main



void bfbcgTest_Case1()
{
    cudaDeviceSynchronize();
    const int M = 5;
    const int K = 5;
    const int N = 3;

    bool debug = false;
    
    // double mtxA_h[] = {        
    //     10.840188, 0.394383, 0.000000, 0.000000, 0.000000,  
    //     0.394383, 10.783099, 0.798440, 0.000000, 0.000000, 
    //     0.000000, 0.798440, 10.911648, 0.197551, 0.000000, 
    //     0.000000, 0.000000, 0.197551, 10.335223, 0.768230, 
    //     0.000000, 0.000000, 0.000000, 0.768230, 10.277775 
    // };

    //Sparse matrix 
    
    int rowOffSets[] = {0, 2, 5, 8, 11, 13};
    int colIndices[] = {0, 1, 0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4};
    double vals[] = {10.840188, 0.394383, 
                      0.394383, 10.783099, 0.798440, 
                      0.798440, 10.911648, 0.197551,
                      0.197551, 10.335223, 0.768230,
                      0.768230, 10.277775};

    int nnz = 13;


    CSRMatrix csrMtxA_h = constructCSRMatrix(M, K, nnz, rowOffSets, colIndices, vals);


    double mtxB_h[] = {-0.957936, 0.099025, -0.312390, -0.141889, 0.429427, 
                    -0.372082, 0.848972, 0.054195, -0.952761, -0.007890,
                    -0.128068, 0.481105, 0.733497, -0.859573, 0.249972};

    // double mtxB_h[K*N];
    // initializeRandom(mtxB_h, K, N);
    // if(debug){
    //     printf("\n\n~~mtxB_h~~\n\n");
    //     print_mtx_clm_h(mtxB_h, K, N);    
    // }


    //(1) Allocate memory
    // double* mtxA_d = NULL;
    double* mtxSolX_d = NULL;
    double* mtxB_d = NULL;

    // CHECK(cudaMalloc((void**)&mtxA_d,  M * K * sizeof(double)));
    CHECK(cudaMalloc((void**)&mtxSolX_d,  K * N * sizeof(double)));
    CHECK(cudaMalloc((void**)&mtxB_d,  M * N * sizeof(double)));

    //(2) Copy value from host to device
    // CHECK(cudaMemcpy(mtxA_d, mtxA_h, M * K * sizeof(double), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(mtxSolX_d, mtxB_h, K * N * sizeof(double), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(mtxB_d, mtxB_h, M * N * sizeof(double), cudaMemcpyHostToDevice));

    if(debug){
        printf("\n\n~~csrMtxA_h~~\n\n");
        print_CSRMtx(csrMtxA_h);
        printf("\n\n~~mtxSolX~~\n\n");
        print_mtx_clm_d(mtxSolX_d, K, N);
        printf("\n\n~~mtxB~~\n\n");
        print_mtx_clm_d(mtxB_d, M, N);
    }

    //Solve AX = B with bfbcg method
    bfbcg(csrMtxA_h, mtxSolX_d, mtxB_d, M, N);

    if(debug){
        printf("\n\n~~📝📝📝Approximate Solution Marix📝📝📝~~\n\n");
        print_mtx_clm_d(mtxSolX_d, K, N);
    }

    //Validate
    //R = B - AX
    printf("\n\n~~🔍👀Validate Solution Matrix X 🔍👀~~");
    double twoNorms = validateBFBCG(csrMtxA_h, M, mtxSolX_d, N, mtxB_d);
    printf("\n\n= = 1st Column Vector 2 norms: %f = =\n\n", twoNorms);


    //()Free memeory
    freeCSRMatrix(csrMtxA_h);
    CHECK(cudaFree(mtxSolX_d));
    CHECK(cudaFree(mtxB_d));



} // end of tranposeTest_Case1

void bfbcgTest_Case2()
{
    cudaDeviceSynchronize();
    bool debug = false;

    const int M = 10;
    const int K = 10;
    const int N = 5;
    
    /*
    SPD Tridiagonal Dense
    10.840188 0.394383 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 
    0.394383 10.783099 0.798440 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 
    0.000000 0.798440 10.911648 0.197551 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 
    0.000000 0.000000 0.197551 10.335223 0.768230 0.000000 0.000000 0.000000 0.000000 0.000000 
    0.000000 0.000000 0.000000 0.768230 10.277775 0.553970 0.000000 0.000000 0.000000 0.000000 
    0.000000 0.000000 0.000000 0.000000 0.553970 10.477397 0.628871 0.000000 0.000000 0.000000 
    0.000000 0.000000 0.000000 0.000000 0.000000 0.628871 10.364784 0.513401 0.000000 0.000000 
    0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.513401 10.952229 0.916195 0.000000 
    0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.916195 10.635712 0.717297 
    0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.717297 10.141603 
    */
    
    
    //Create Sample case to set up
    // CSRMatrix csrMtxA_h =generateSparseSPDMatrixCSR(M);
    // print_CSRMtx(csrMtxA_h);
    // double* mtxA_h = csrToDense(csrMtxA_h);
    // print_mtx_clm_h(mtxA_h, M, M);
    
    //Sparse matrix 
    int rowOffSets[] = {0, 2, 5, 8, 11, 14, 17, 20, 23, 26, 28};
    int colIndices[] = {0, 1, 0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4, 5, 4, 5, 6, 5, 6, 7, 6, 7, 8, 7, 8, 9, 8, 9};
    double vals[] = {10.840188, 0.394383,
                    0.394383, 10.783099, 0.798440,
                    0.798440, 10.911648, 0.197551,
                    0.197551, 10.335223, 0.768230,
                    0.768230, 10.277775, 0.553970,
                    0.553970, 10.477397, 0.628871, 
                    0.628871, 10.364784, 0.513401, 
                    0.513401, 10.952229, 0.916195,
                    0.916195, 10.635712, 0.717297,
                    0.717297, 10.141603
                    };

    int nnz = 28;


    CSRMatrix csrMtxA_h = constructCSRMatrix(M, K, nnz, rowOffSets, colIndices, vals);
    
    
    double mtxB_h[] = {-0.917206, 0.742276, 0.899125, -0.558456, 0.726547,0.343939, -0.319879, -0.901800, 0.898783,-0.885493,
                    -0.436044,-0.657422, -0.713946, -0.201146, -0.017785, -0.066716,-0.708516, 0.551868, 0.520759, -0.482016,
                    0.951020, -0.664174, 0.354639, 0.692534, 0.972838, 0.722408, -0.002730, -0.239378, 0.006314, -0.667044,
                    0.285601, 0.089107, -0.924769, 0.184726,0.530651, 0.801779, -0.471335, -0.789228, 0.899979,-0.572552,
                    -0.674721, -0.536065, -0.229974, -0.388667,0.262788, 0.752241, 0.544616, 0.554272, 0.304109, 0.065375};
     
    // double mtxB_h[K*N];
    // initializeRandom(mtxB_h, K, N);
    // if(debug){
    //     printf("\n\n~~mtxB_h~~\n\n");
    //     print_mtx_clm_h(mtxB_h, K, N);    
    // }

    //(1) Allocate memory
    double* mtxSolX_d = NULL;
    double* mtxB_d = NULL;

    CHECK(cudaMalloc((void**)&mtxSolX_d,  K * N * sizeof(double)));
    CHECK(cudaMalloc((void**)&mtxB_d,  M * N * sizeof(double)));

    //(2) Copy value from host to device
    CHECK(cudaMemcpy(mtxSolX_d, mtxB_h, K * N * sizeof(double), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(mtxB_d, mtxB_h, M * N * sizeof(double), cudaMemcpyHostToDevice));

    if(debug){
        printf("\n\n~~csrMtxA_h~~\n\n");
        print_CSRMtx(csrMtxA_h);
        printf("\n\n~~mtxSolX~~\n\n");
        print_mtx_clm_d(mtxSolX_d, K, N);
        printf("\n\n~~mtxB~~\n\n");
        print_mtx_clm_d(mtxB_d, M, N);
    }

    //Solve AX = B with bfbcg method
    bfbcg(csrMtxA_h, mtxSolX_d, mtxB_d, M, N);

    if(debug){
        printf("\n\n~~📝📝📝Approximate Solution Marix📝📝📝~~\n\n");
        print_mtx_clm_d(mtxSolX_d, K, N);
    }

    //Validate
    //R = B - AX
    printf("\n\n\n\n🔍👀Validate Solution Matrix X 🔍👀");
    double twoNorms = validateBFBCG(csrMtxA_h, M, mtxSolX_d, N, mtxB_d);
    printf("\n\n~~~ mtxR = B - AX_sol, 1st Column Vector 2 norms in mtxR : %f ~~~\n\n", twoNorms);


    // //()Free memeory
    // free(mtxA_h); // Generate Sample Case
    freeCSRMatrix(csrMtxA_h);
    CHECK(cudaFree(mtxSolX_d));
    CHECK(cudaFree(mtxB_d));

} // end of tranposeTest_Case2

void bfbcgTest_Case3()
{
    cudaDeviceSynchronize();
    bool debug = false;

    const int M = 16;
    const int K = 16;
    const int N = 15;
    
    /*
    SPD Tridiagonal Dense
    10.840188 0.394383 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 
    0.394383 10.783099 0.798440 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 
    0.000000 0.798440 10.911648 0.197551 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 
    0.000000 0.000000 0.197551 10.335223 0.768230 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 
    0.000000 0.000000 0.000000 0.768230 10.277775 0.553970 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 
    0.000000 0.000000 0.000000 0.000000 0.553970 10.477397 0.628871 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 
    0.000000 0.000000 0.000000 0.000000 0.000000 0.628871 10.364784 0.513401 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 
    0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.513401 10.952229 0.916195 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 
    0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.916195 10.635712 0.717297 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 
    0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.717297 10.141603 0.606969 0.000000 0.000000 0.000000 0.000000 0.000000 
    0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.606969 10.016300 0.242887 0.000000 0.000000 0.000000 0.000000 
    0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.242887 10.137232 0.804177 0.000000 0.000000 0.000000 
    0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.804177 10.156679 0.400944 0.000000 0.000000 
    0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.400944 10.129790 0.108809 0.000000 
    0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.108809 10.998924 0.218257 
    0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.218257 10.512933 
    */
    
    
    // //Create Sample case to set up
    // CSRMatrix csrMtxA_h =generateSparseSPDMatrixCSR(M);
    // print_CSRMtx(csrMtxA_h);
    // double* mtxA_h = csrToDense(csrMtxA_h);
    // print_mtx_clm_h(mtxA_h, M, M);
    
    //Sparse matrix 
    int rowOffSets[] = {0, 2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35, 38, 41, 44, 46};
    int colIndices[] = {0, 1, 0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4, 5, 4, 5, 6, 5, 6, 7, 6, 7, 8, 7, 8, 9, 8, 9, 10, 9, 10, 11, 10, 11, 12, 11, 12, 13, 12, 13, 14, 13, 14, 15, 14, 15};
    double vals[] = {10.840188, 0.394383,
                    0.394383, 10.783099, 0.798440,
                    0.798440, 10.911648, 0.197551, 
                    0.197551, 10.335223, 0.768230, 
                    0.768230, 10.277775, 0.553970,
                    0.553970, 10.477397, 0.628871,
                    0.628871, 10.364784, 0.513401, 
                    0.513401, 10.952229, 0.916195, 
                    0.916195, 10.635712, 0.717297, 
                    0.717297, 10.141603, 0.606969,
                    0.606969, 10.016300, 0.242887,
                    0.242887, 10.137232, 0.804177,
                    0.804177, 10.156679, 0.400944, 
                    0.400944, 10.129790, 0.108809, 
                    0.108809, 10.998924, 0.218257, 
                    0.218257, 10.512933
                    };

    int nnz = 46;


    CSRMatrix csrMtxA_h = constructCSRMatrix(M, K, nnz, rowOffSets, colIndices, vals);
    
    double mtxB_h[] = {
        -0.828583, -0.206065, -0.357603, 0.477982, -0.268613, -0.063139, 0.296667, -0.326744, 0.445940, -0.912228, 0.211429, 0.346067, 0.664094, 0.893560, 0.828269, -0.867125,
        -0.394889, 0.143599, -0.608832, 0.495604, -0.445249, 0.363128, 0.031231, -0.144173, -0.497560, 0.282189, 0.928123, 0.864081, 0.645952, 0.110728, -0.585657, 0.817369,
        0.904663, 0.056740, 0.295351, -0.363950, 0.993600, -0.407982, 0.309307, 0.439540, -0.320210, -0.479264, -0.214393, -0.656116, -0.585704, -0.386123, -0.523241, 0.019407,
        0.757476, -0.132072, -0.484989, -0.687773, -0.768944, 0.546242, 0.168054, -0.266504, -0.171569, 0.096177, -0.402423, -0.525617, -0.793095, 0.011920, -0.708248, -0.888432,
        -0.931341, 0.587103, -0.252382, -0.937740, -0.820880, -0.943076, 0.501800, -0.141089, -0.422340, -0.712593, 0.202795, -0.008043, -0.098716, 0.679554, -0.988636, -0.341239,
        -0.452518, -0.473626, -0.029012, -0.221463, -0.927384, -0.860959, 0.512033, -0.098953, 0.235218, -0.890390, 0.375430, 0.442123, 0.121529, 0.667182, 0.553691, 0.190189,
        0.254284, -0.698691, 0.252448, 0.433405, -0.641767, -0.245752, -0.707684, -0.064107, 0.041656, 0.495110, 0.927850, 0.942940, 0.174664, 0.939214, -0.398299, 0.722146,
        -0.534412, 0.572688, -0.499316, -0.461795, 0.711730, -0.987283, 0.439252, -0.053052, -0.877674, -0.185318, -0.610929, 0.243856, -0.518136, 0.942762, -0.565955, 0.736148,
        -0.755929, 0.686493, 0.169553, -0.397696, -0.559259, 0.461869, 0.538197, 0.482397, -0.043021, 0.466047, 0.425337, -0.868356, 0.405261, -0.972962, 0.853790, 0.870849,
        0.599726, -0.645527, -0.590946, 0.311456, -0.632810, 0.848306, -0.741596, -0.510484, -0.337012, -0.352525, 0.733372, 0.144852, -0.409763, -0.832583, -0.118999, -0.165692,
        0.853910, -0.949446, 0.436612, -0.705348, 0.512423, -0.025191, 0.777049, -0.530598, -0.559144, 0.202386, -0.398954, 0.846117, 0.229424, -0.545165, 0.716966, -0.170850,
        -0.190692, -0.873980, -0.859394, 0.176498, 0.974326, -0.600990, 0.666015, -0.362686, 0.046485, 0.399387, 0.782166, 0.636723, 0.566804, -0.336833, -0.528969, 0.420714,
        -0.286279, 0.907642, 0.715365, -0.773856, -0.117549, 0.492414, -0.304454, 0.323307, -0.305200, 0.296592, 0.169424, 0.924224, 0.751427, -0.113610, -0.246626, -0.439265,
        0.012410, -0.106020, 0.737234, -0.013265, 0.292990, 0.403248, 0.624049, -0.660524, -0.197365, 0.406215, 0.976199, -0.630561, -0.930618, -0.552771, 0.790152, -0.216897,
        -0.645129, 0.505517, 0.009247, 0.237322, -0.002069, 0.704793, -0.439371, 0.692731, 0.001385, 0.730053, 0.616955, -0.247188, -0.383557, -0.629671, 0.313547, 0.628853
    };


    // double mtxB_h[K*N];
    // initializeRandom(mtxB_h, K, N);
    // if(debug){
    //     printf("\n\n~~mtxB_h~~\n\n");
    //     print_mtx_clm_h(mtxB_h, K, N);    
    // }
    

    //(1) Allocate memory
    double* mtxSolX_d = NULL;
    double* mtxB_d = NULL;

    CHECK(cudaMalloc((void**)&mtxSolX_d,  K * N * sizeof(double)));
    CHECK(cudaMalloc((void**)&mtxB_d,  M * N * sizeof(double)));

    //(2) Copy value from host to device
    CHECK(cudaMemcpy(mtxSolX_d, mtxB_h, K * N * sizeof(double), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(mtxB_d, mtxB_h, M * N * sizeof(double), cudaMemcpyHostToDevice));

    if(debug){
        printf("\n\n~~csrMtxA_h~~\n\n");
        print_CSRMtx(csrMtxA_h);
        printf("\n\n~~mtxSolX~~\n\n");
        print_mtx_clm_d(mtxSolX_d, K, N);
        printf("\n\n~~mtxB~~\n\n");
        print_mtx_clm_d(mtxB_d, M, N);
    }

    //Solve AX = B with bfbcg method
    bfbcg(csrMtxA_h, mtxSolX_d, mtxB_d, M, N);

    if(debug){
        printf("\n\n~~📝📝📝Approximate Solution Marix📝📝📝~~\n\n");
        print_mtx_clm_d(mtxSolX_d, K, N);
    }

    //Validate
    //R = B - AX
    printf("\n\n\n\n🔍👀Validate Solution Matrix X 🔍👀");
    double twoNorms = validateBFBCG(csrMtxA_h, M, mtxSolX_d, N, mtxB_d);
    printf("\n\n~~~ mtxR = B - AX_sol, 1st Column Vector 2 norms in mtxR : %f ~~~\n\n", twoNorms);


    // //()Free memeory
    // free(mtxA_h); // Generate Sample Case
    freeCSRMatrix(csrMtxA_h);
    CHECK(cudaFree(mtxSolX_d));
    CHECK(cudaFree(mtxB_d));

} // end of tranposeTest_Case3


void bfbcgTest_Case4()
{
    cudaDeviceSynchronize();
    bool debug = false;

    const int M = 32768; // 2^15
    const int K = 32768;
    const int N = 16;
    

    CSRMatrix csrMtxA_h = generateSparseSPDMatrixCSR(M);
    

    double mtxB_h[K*N];
    initializeRandom(mtxB_h, K, N);
    

    //(1) Allocate memory
    double* mtxSolX_d = NULL;
    double* mtxB_d = NULL;

    CHECK(cudaMalloc((void**)&mtxSolX_d,  K * N * sizeof(double)));
    CHECK(cudaMalloc((void**)&mtxB_d,  M * N * sizeof(double)));

    //(2) Copy value from host to device
    CHECK(cudaMemcpy(mtxSolX_d, mtxB_h, K * N * sizeof(double), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(mtxB_d, mtxB_h, M * N * sizeof(double), cudaMemcpyHostToDevice));

    if(debug){
        printf("\n\n~~csrMtxA_h~~\n\n");
        print_CSRMtx(csrMtxA_h);
        printf("\n\n~~mtxSolX~~\n\n");
        print_mtx_clm_d(mtxSolX_d, K, N);
        printf("\n\n~~mtxB~~\n\n");
        print_mtx_clm_d(mtxB_d, M, N);
    }

    //Solve AX = B with bfbcg method
    bfbcg(csrMtxA_h, mtxSolX_d, mtxB_d, M, N);

    if(debug){
        printf("\n\n~~📝📝📝Approximate Solution Marix📝📝📝~~\n\n");
        print_mtx_clm_d(mtxSolX_d, K, N);
    }

    //Validate
    //R = B - AX
    printf("\n\n\n\n🔍👀Validate Solution Matrix X 🔍👀");
    double twoNorms = validateBFBCG(csrMtxA_h, M, mtxSolX_d, N, mtxB_d);
    printf("\n\n~~~ mtxR = B - AX_sol, 1st Column Vector 2 norms in mtxR : %f ~~~\n\n", twoNorms);


    // //()Free memeory
    // free(mtxA_h); // Generate Sample Case
    freeCSRMatrix(csrMtxA_h);
    CHECK(cudaFree(mtxSolX_d));
    CHECK(cudaFree(mtxB_d));

} // end of tranposeTest_Case4




/*
Sample Run
= = = =bfbcgTest.cu= = = = 



🔍🔍🔍 Test Case 1 🔍🔍🔍




💫💫💫 Iteration 1 💫💫💫 

= = current Rank: 3 = =

Q <- AP: 0.000060 s 

Alpha <- (P'Q)^{-1} * (P'R): 0.007182 s 

X_{i+1} <- x_{i} + P * alpha: 0.000004 s 

R_{i+1} <- R_{i} - Q * alpha: 0.000005 s 

🫥Relative Residue: 0.040127🫥


Z_{i+1} <- MR_{i+1}: 0.000052 s 

beta <- -(P'Q)^{-1} * (Q'Z_{i+1}): 0.000011 s 

P_{i+1} = orth(Z_{i+1} + p * beta): 0.001928 s 



💫💫💫 Iteration 2 💫💫💫 

= = current Rank: 2 = =

🫥Relative Residue: 0.000000🫥



🌀🌀🌀CONVERGED🌀🌀🌀



~~🔍👀Validate Solution Matrix X 🔍👀~~

= = 1st Column Vector 2 norms: 0.000000 = =



🔍🔍🔍 Test Case 2 🔍🔍🔍




💫💫💫 Iteration 1 💫💫💫 

= = current Rank: 5 = =

Q <- AP: 0.000060 s 

Alpha <- (P'Q)^{-1} * (P'R): 0.001926 s 

X_{i+1} <- x_{i} + P * alpha: 0.000004 s 

R_{i+1} <- R_{i} - Q * alpha: 0.000004 s 

🫥Relative Residue: 0.054932🫥


Z_{i+1} <- MR_{i+1}: 0.000049 s 

beta <- -(P'Q)^{-1} * (Q'Z_{i+1}): 0.000010 s 

P_{i+1} = orth(Z_{i+1} + p * beta): 0.002170 s 



💫💫💫 Iteration 2 💫💫💫 

= = current Rank: 5 = =

🫥Relative Residue: 0.000000🫥



🌀🌀🌀CONVERGED🌀🌀🌀





🔍👀Validate Solution Matrix X 🔍👀

~~~ mtxR = B - AX_sol, 1st Column Vector 2 norms in mtxR : 0.000000 ~~~



🔍🔍🔍 Test Case 3 🔍🔍🔍




💫💫💫 Iteration 1 💫💫💫 

= = current Rank: 15 = =

Q <- AP: 0.000068 s 

Alpha <- (P'Q)^{-1} * (P'R): 0.002644 s 

X_{i+1} <- x_{i} + P * alpha: 0.000005 s 

R_{i+1} <- R_{i} - Q * alpha: 0.000004 s 

🫥Relative Residue: 0.004301🫥


Z_{i+1} <- MR_{i+1}: 0.000057 s 

beta <- -(P'Q)^{-1} * (Q'Z_{i+1}): 0.000011 s 

P_{i+1} = orth(Z_{i+1} + p * beta): 0.002468 s 



💫💫💫 Iteration 2 💫💫💫 

= = current Rank: 1 = =

🫥Relative Residue: 0.000000🫥



🌀🌀🌀CONVERGED🌀🌀🌀





🔍👀Validate Solution Matrix X 🔍👀

~~~ mtxR = B - AX_sol, 1st Column Vector 2 norms in mtxR : 0.000000 ~~~



🔍🔍🔍 Test Case 4 🔍🔍🔍




💫💫💫 Iteration 1 💫💫💫 

= = current Rank: 16 = =

Q <- AP: 0.000295 s 

Alpha <- (P'Q)^{-1} * (P'R): 0.002734 s 

X_{i+1} <- x_{i} + P * alpha: 0.000006 s 

R_{i+1} <- R_{i} - Q * alpha: 0.000004 s 

🫥Relative Residue: 0.080048🫥


Z_{i+1} <- MR_{i+1}: 0.000130 s 

beta <- -(P'Q)^{-1} * (Q'Z_{i+1}): 0.000016 s 

P_{i+1} = orth(Z_{i+1} + p * beta): 0.004892 s 



💫💫💫 Iteration 2 💫💫💫 

= = current Rank: 16 = =

🫥Relative Residue: 0.006311🫥




💫💫💫 Iteration 3 💫💫💫 

= = current Rank: 16 = =

🫥Relative Residue: 0.000530🫥




💫💫💫 Iteration 4 💫💫💫 

= = current Rank: 16 = =

🫥Relative Residue: 0.000047🫥




💫💫💫 Iteration 5 💫💫💫 

= = current Rank: 16 = =

🫥Relative Residue: 0.000004🫥




💫💫💫 Iteration 6 💫💫💫 

= = current Rank: 16 = =

🫥Relative Residue: 0.000000🫥



!!!Current Rank became 0!!!
 🔸Exit iteration🔸




🔍👀Validate Solution Matrix X 🔍👀

~~~ mtxR = B - AX_sol, 1st Column Vector 2 norms in mtxR : 0.000392 ~~~



= = = = end of bfbcgTest = = = =
*/