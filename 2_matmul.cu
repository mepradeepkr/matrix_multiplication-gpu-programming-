#include<cuda.h>
#include<cuda_runtime.h>
#include<stdio.h>
#define N 512
#define BLOCK_SIZE 32

__global__ void matrixMul( int * , int * , int * , int );

__global__ 
void matrixMul(int  *dev_A, int  *dev_B, int  *matOut, int  n){

    int  row = blockDim.y * blockIdx.y + threadIdx.y;
    int  col = blockDim.x * blockIdx.x + threadIdx.x;
    

    //############## MATRIX MULTIPLICATION ###############
    int  sum = 0;
    if(row < n && col < n){
        for(int  i = 0; i < n; i++)
            sum +=  dev_A[row*n + i] * dev_B[n*i + col];
        matOut[row * n + col] = sum;
    }

    //################# MATRIX ADDITION ####################
    //matOut[row * n + col] = dev_A[row * n + col] + dev_B[row * n + col];
}

int  main(){
    int  n = N;
    int  size = n*n*sizeof(int );

    int  host_A[N][N], host_B[N][N], host_C[N][N];
    
    int  *dev_A, *dev_B, *dev_C;

    cudaMalloc((void**)&dev_A, size);
    cudaMalloc((void**)&dev_B, size);
    cudaMalloc((void**)&dev_C, size);

    for(int  i=0;i<n; i++){
        for(int  j=0; j<n; j++){
            host_A[i][j] = 2;
            host_B[i][j] = 2;
        }
    }

    cudaMemcpy(dev_A, host_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_B, host_B, size, cudaMemcpyHostToDevice);

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(ceil(n/BLOCK_SIZE), ceil(n/BLOCK_SIZE));

    for(int  i = 0; i < N ; i += 1024){
        
    }
    matrixMul<<<grid, block>>>(dev_A, dev_B, dev_C, n);
    

    cudaMemcpy(host_C, dev_C, size, cudaMemcpyDeviceToHost);


    for(int  i=0;i<n; i++){
        for(int  j=0; j<n; j++){
            if(host_C[i][j] != 4*512){
                printf("Error @ idx[%d][%d] val = %d \n", i, j, host_C[i][j]);
            }
        }
    }
    printf("val = %d\n", host_C[23][31]);
    return(0);
}


