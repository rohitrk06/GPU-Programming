#include <stdio.h>
#include <cuda.h>

#define N 6

__global__ void vecAdd(float *a, float *b, float *c, int n){
    int id = threadIdx.x;
    if (id < n){
        c[id] = a[id] + b[id];
    }
}

int main(){
    int size = N * sizeof(int);
    float *a_h, *b_h;
    
    a_h = (float *)malloc(size);
    b_h = (float *) malloc(size);

    FILE *file = fopen("input_vecAdd", "r");

    for(int ii = 0; ii < N; ++ii){
        fscanf( file ,"%f %f", a_h + ii, b_h + ii);
    }
    fclose(file);
    float *a_d, *b_d, *c_d;
    cudaMalloc(&a_d, size);
    cudaMalloc(&b_d, size);
    cudaMalloc(&c_d, size);

    cudaMemcpy(a_d, a_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b_h, size, cudaMemcpyHostToDevice);

    free(b_h);

    vecAdd<<<1, N>>>(a_d, b_d, c_d, N);

    cudaMemcpy(a_h, c_d, size, cudaMemcpyDeviceToHost);

    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);

    for (int ii = 0; ii < N; ++ii){
        printf("%.2f ", a_h[ii]);
    }
    printf("\n");

    free(a_h);

}