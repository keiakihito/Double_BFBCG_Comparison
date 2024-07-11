#ifndef CSRMatrix_H
#define CSRMatrix_H

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cusparse_v2.h>
#include <time.h>

#define CHECK(call){ \
    const cudaError_t cuda_ret = call; \
    if(cuda_ret != cudaSuccess){ \
        printf("Error: %s:%d,  ", __FILE__, __LINE__ );\
        printf("code: %d, reason: %s \n", cuda_ret, cudaGetErrorString(cuda_ret));\
        exit(-1); \
    }\
}

// CSR Matrix
struct CSRMatrix{
    int numOfRows;
    int numOfClms;
    int numOfnz;
    int *row_offsets;
    int *col_indices;
    float *vals;
};


// Generate a random tridiagonal symmetric matrix
//It comes from CUDA CG sample code to generate sparse tridiagobal matrix
void genTridiag(int *I, int *J, float *val, int N, int nz) {
    I[0] = 0;
    J[0] = 0;
    J[1] = 1;
    val[0] = (float)rand() / RAND_MAX + 10.0f;
    val[1] = (float)rand() / RAND_MAX;
    int start;

    for (int i = 1; i < N; i++) {
        if (i > 1) {
            I[i] = I[i - 1] + 3;
        } else {
            I[1] = 2;
        }

        start = (i - 1) * 3 + 2;
        J[start] = i - 1;
        J[start + 1] = i;

        if (i < N - 1) {
            J[start + 2] = i + 1;
        }

        val[start] = val[start - 1];
        val[start + 1] = (float)rand() / RAND_MAX + 10.0f;

        if (i < N - 1) {
            val[start + 2] = (float)rand() / RAND_MAX;
        }
    }

    I[N] = nz;
}

// Generate a sparse SPD matrix in CSR format
CSRMatrix generateSparseSPDMatrixCSR(int N) {
    int nzMax = 3 * N - 2; // Maximum non-zero elements for a tridiagonal matrix
    int *row_offsets = (int*)malloc((N + 1) * sizeof(int));
    int *col_indices = (int*)malloc(nzMax * sizeof(int));
    float *vals = (float*)malloc(nzMax * sizeof(float));

    genTridiag(row_offsets, col_indices, vals, N, nzMax);

    // Create CSRMatrix object with the result
    CSRMatrix csrMtx;
    csrMtx.numOfRows = N;
    csrMtx.numOfClms = N;
    csrMtx.numOfnz = nzMax;
    csrMtx.row_offsets = row_offsets;
    csrMtx.col_indices = col_indices;
    csrMtx.vals = vals;

    return csrMtx;
}

// Generate a sparse SPD matrix in CSR format
CSRMatrix generateSparseIdentityMatrixCSR(int N) {
    int *row_offsets = (int*)malloc((N + 1) * sizeof(int));
    int *col_indices = (int*)malloc(N * sizeof(int));
    float *vals = (float*)malloc(N * sizeof(float));

    if (!row_offsets || !col_indices || !vals) {
        fprintf(stderr, "\n\nFailed to allocate memory for CSR matrix. \n\n");
        exit(EXIT_FAILURE);
    }


    // Fill row_offsets, col_indices, and vals
    for (int wkr = 0; wkr < N; ++wkr) {
        row_offsets[wkr] = wkr;
        col_indices[wkr] = wkr;
        vals[wkr] = 1.0f;
    }

    // Last element of row_offsets should be the number of non-zero elements
    row_offsets[N] = N; 


    // Create CSRMatrix object with the result
    CSRMatrix csrMtx;
    csrMtx.numOfRows = N;
    csrMtx.numOfClms = N;
    csrMtx.numOfnz = N;
    csrMtx.row_offsets = row_offsets;
    csrMtx.col_indices = col_indices;
    csrMtx.vals = vals;

    return csrMtx;
}


void freeCSRMatrix(CSRMatrix &csrMtx){
    free(csrMtx.row_offsets);
    free(csrMtx.col_indices);
    free(csrMtx.vals);
} // end of freeCSRMatrix


float* csrToDense(const CSRMatrix &csrMtx)
{
    float *dnsMtx = (float*)calloc(csrMtx.numOfRows * csrMtx.numOfClms, sizeof(float));
    for(int otWkr = 0; otWkr < csrMtx.numOfRows; otWkr++){
        for(int inWkr = csrMtx.row_offsets[otWkr]; inWkr < csrMtx.row_offsets[otWkr+1]; inWkr++){
            dnsMtx[otWkr * csrMtx.numOfClms + csrMtx.col_indices[inWkr]] = csrMtx.vals[inWkr];
        }// end of inner loop
    } // end of outer loop

    return dnsMtx;

} // end of csrToDense


//Print out CSRMatrix object
void print_CSRMtx(const CSRMatrix &csrMtx)
{
    printf("\n\nnumOfRows: %d, numOfClms: %d , number of non zero: %d", csrMtx.numOfRows, csrMtx.numOfClms, csrMtx.numOfnz);

    printf("\n\nrow_offsets: ");
    for(int wkr = 0; wkr <= csrMtx.numOfRows; wkr++){
        if(wkr == 0){
            printf("\n[ ");
        }
        printf("%d ", csrMtx.row_offsets[wkr]);
        if(wkr == csrMtx.numOfRows){
            printf("]\n");
        }
    }

    printf("\n\ncol_indices: ");
    for(int wkr = 0; wkr < csrMtx.numOfnz; wkr++){
        if(wkr == 0){
            printf("\n[ ");
        }
        printf("%d ", csrMtx.col_indices[wkr]);
        if(wkr == csrMtx.numOfnz - 1){
            printf("]\n");
        }
    }

    printf("\n\nnon zero values: ");
    for(int wkr = 0; wkr < csrMtx.numOfnz; wkr++){
        if(wkr == 0){
            printf("\n[ ");
        }
        printf("%f ", csrMtx.vals[wkr]);
        if(wkr == csrMtx.numOfnz - 1){
            printf("]\n");
        }
    }

    printf("\n");

} // end of print_CSRMtx




#endif // CSRMatix_h