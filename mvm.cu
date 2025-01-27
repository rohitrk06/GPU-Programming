#include <cuda.h>
#include <stdio.h>

#define blockSize 1024

__global__ void matrixVectorMul(unsigned *mat, unsigned *vec, unsigned *result, int matDim_m , int matDim_n){
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id<matDim_m){
        unsigned sum =0;
        for (int jj = 0; jj < matDim_n; ++jj){
            sum += mat[id* matDim_n + jj] * vec[jj];
        }
        result[id] = sum;
    }
}

int main(){
    unsigned int *h_mat, *h_vector, *h_result;

    FILE *inputMatrix, *inputVector;

    if ((inputMatrix = fopen("inputMatrix_mvm","r")) == NULL) {
        printf("Failed to open the file inputMatrix_mvm");
        return 1;
    }

    if ((inputVector = fopen("inputVector_mvm","r"))==NULL){
        printf("Failed to open the file inputVector_mvm");
        return 1;
    }

    int matDim_m, matDim_n, vecDim;
    fscanf(inputMatrix, "%d %d", &matDim_m, &matDim_n);
    fscanf(inputVector, "%d", &vecDim);

    if (matDim_n != vecDim){
        printf("Matrix and Vector dimensions do not match\n");
        return 1;
    }

    h_mat = (unsigned int *)malloc(matDim_m * matDim_n * sizeof(unsigned int));
    h_vector = (unsigned int *)malloc(vecDim * sizeof(unsigned int));
    h_result = (unsigned int *)malloc(matDim_m * sizeof(unsigned int));

    int counter = 0, row = 0;
    while (fscanf(inputMatrix, "%d" , h_mat + (row * matDim_n + counter)) != EOF){
        counter++;
        if (counter%matDim_n == 0) {
            counter = 0;
            row++;
        }
    }

    counter = 0;
    while(fscanf(inputVector, "%d", h_vector + counter) != EOF){
        counter++;
    }

    // Allocating memory on the GPU
    unsigned int *d_mat, *d_vector, *d_result;

    cudaError_t err = cudaMalloc(&d_mat, matDim_m * matDim_n * sizeof(unsigned int));
    if (err != cudaSuccess){
        printf("Error: %s\n In %s at line %d", cudaGetErrorString(err), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }

    err = cudaMalloc(&d_vector, vecDim * sizeof(unsigned int));
    if (err != cudaSuccess){
        printf("Error: %s\n In %s at line %d", cudaGetErrorString(err), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }

    err = cudaMalloc(&d_result, matDim_m * sizeof(unsigned int));
    if (err != cudaSuccess){
        printf("Error: %s\n In %s at line %d", cudaGetErrorString(err), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }

    cudaMemcpy(d_mat,h_mat,matDim_m * matDim_n * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vector, h_vector, vecDim * sizeof(unsigned int), cudaMemcpyHostToDevice);

    free(h_mat);
    free(h_vector);

    // Kernel Launch

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int noBlock = ceil((matDim_m)/(float) blockSize);

    cudaEventRecord(start,0);
    matrixVectorMul<<<noBlock, blockSize>>>(d_mat, d_vector, d_result, matDim_m, matDim_n);
    cudaEventRecord(stop,0);
    cudaMemcpy(h_result, d_result, matDim_m * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaEventSynchronize(stop);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Time taken for matrix vector multiplication is %f ms\n", elapsedTime);

    cudaFree(d_mat);
    cudaFree(d_vector);
    cudaFree(d_result);

    printf("Resultant Vector\n");
    for (int ii = 0; ii < matDim_m; ++ii){
        printf("%d\n", h_result[ii]);
    }
    free(h_result);
    return 0;
    
}