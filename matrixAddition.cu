#include <stdio.h>
#include <cuda.h>

#define blockSize 1024

__global__ void matrixAddKernal(int *A, int *B, int *C, int m, int n){
    // This is the matrix addition kernel that handles one matrix element per thread
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id < m*n){
        C[id] = A[id] + B[id];
    }
}

__global__ void matrixAddKernal_v1(int *A, int *B, int *C, int m, int n){
    // This is the matrix addition kernel that handles one row per thread
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id < m){
        for (int jj = 0; jj < n; ++jj){
            C[id*n + jj] = A[id*n + jj] + B[id*n + jj];
        }
    }
}

__global__ void matrixAddKernal_v2(int *A, int *B, int *C, int m, int n){
    // This is the matrix addition kernel that handles one column per thread
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id < n){
        for (int ii =0; ii < m; ++ii){
            C[ii*n + id] = A[ii*n + id] + B[ii*n + id];
        }
    }
}

int main(){
    
    // Matrix Dimension
    int m , n;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    scanf("%d %d", &m, &n);

    int *h_A, *h_B, *h_C;

    h_A = (int *)malloc(m*n*sizeof(int));
    h_B = (int *)malloc(m*n*sizeof(int));
    h_C = (int *)malloc(m*n*sizeof(int));

    for (int ii =0 ; ii < m; ++ii){
        for (int jj = 0; jj < n; ++jj){
            scanf("%d", h_A + ((ii*n) + jj));
        }
    }

    for (int ii =0 ; ii < m; ++ii){
        for (int jj = 0; jj < n; ++jj){
            scanf("%d", h_B + ((ii*n) + jj));
        }
    }

    // Matrix memory allocation on device
    int *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, m*n*sizeof(int));
    cudaMemcpy(d_A, h_A, m*n*sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&d_B, m*n*sizeof(int));
    cudaMemcpy(d_B, h_B, m*n*sizeof(int), cudaMemcpyHostToDevice);

    free(h_A);
    free(h_B);

    cudaMalloc(&d_C, m*n*sizeof(int));

    // Kernel Launch
    int noBlocks = ceil((n*m)/(float) blockSize);
    cudaEventRecord(start,0);
    // matrixAddKernal<<<noBlocks, blockSize>>>(d_A, d_B, d_C, m, n);
    matrixAddKernal_v1<<<noBlocks, blockSize>>>(d_A, d_B, d_C, m, n);
    // matrixAddKernal_v2<<<noBlocks, blockSize>>>(d_A, d_B, d_C, m, n);
    cudaEventRecord(stop,0);
    cudaMemcpy(h_C, d_C, m*n*sizeof(int), cudaMemcpyDeviceToHost);
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Time taken for matrix addition is %f ms\n", elapsedTime);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    printf("Matrix A\n");
    for (int ii =0 ; ii < m; ++ii){
       for (int jj = 0; jj < n; ++jj){
           printf("%d ", h_C[(ii*n) + jj]);
       }
       printf("\n");
    }
    free(h_C);
}