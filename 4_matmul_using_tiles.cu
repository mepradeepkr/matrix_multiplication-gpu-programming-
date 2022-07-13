#include<cuda.h>
#include<cuda_runtime.h>
#include<stdio.h>
#define WIDTH 512
#define BLOCK_SIZE 8
#define TILE_WIDTH 8

__global__ void matrixMul( int * , int * , int * , int );

__global__ 
void matrixMul(int  *dev_A, int  *dev_B, int  *matOut, int  n){
    __shared__ int A_sh[TILE_WIDTH][TILE_WIDTH];
    __shared__ int B_sh[TILE_WIDTH][TILE_WIDTH];
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * blockDim.y + ty;
    int col = bx * blockDim.x + tx;

    int pSum=0;
    
    for(int i=0; i < WIDTH/TILE_WIDTH; i++){
        // Threads are loading the data simulteniously in shared memory
        A_sh[ty][tx] = dev_A[row * WIDTH + i * TILE_WIDTH + tx];
        B_sh[ty][tx] = dev_B[(i * TILE_WIDTH + ty)* WIDTH + col];
        __syncthreads();

        for(int k=0; k < TILE_WIDTH; k++){
            pSum += A_sh[ty][k] * B_sh[k][tx]; 
        }
        __syncthreads();
    }
    matOut[row*WIDTH + col] = pSum; 
}

int  main(){
    int  n = WIDTH;
    int  size = n*n*sizeof(int );

    int  host_A[WIDTH][WIDTH], host_B[WIDTH][WIDTH], host_C[WIDTH][WIDTH];
    
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

    dim3 block(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 grid(ceil(n/BLOCK_SIZE), ceil(n/BLOCK_SIZE), 1);

    for(int  i = 0; i < WIDTH ; i += 1024){
        
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


