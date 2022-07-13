#include<cuda.h>
#include<cuda_runtime.h>
#include<stdio.h>
//#include<vector_sum.cu>

#define ARR_SIZE 64

__global__ void vectorAdd(float*, float*, float*, int); 

__global__
void vectorAdd(float* A, float* B, float*C, int n){
    int row = blockDim.x*blockIdx.x + threadIdx.x;
    int col = blockDim.y*blockIdx.y + threadIdx.y;

    for(int i=0; i<n; i++){
        //C[row, col] = A[row, i] * B[i, col]
        C[row * n + col] += A[row * n + i] * B[n*i + col];
    }
}

void vecAdd(float* h_A, float* h_B, float* h_C, int n){
    //to find out num of threads
    int deviceID;
    cudaDeviceProp props;
    cudaGetDevice(&deviceID);
    cudaGetDeviceProperties(&props, deviceID);

    int max_threads_per_block = props.maxThreadsPerBlock;
    printf("\nMaximum number of threads per block : %d\n", max_threads_per_block);

    int size = n*n*sizeof(float);
    
    float *d_A=NULL, *d_B=NULL, *d_C=NULL;

    //Error code to check return values for cuda cells
    cudaError_t err = cudaSuccess;

    err = cudaMalloc((void**)&d_A, size);
    
    if(err != cudaSuccess){
        fprintf(stderr, "Failed to allocate device vector B(error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMalloc((void **)&d_B, size);
    
    if(err != cudaSuccess){
        fprintf(stderr, "Failed to allocate device vector B(error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMalloc((void **)&d_C, size);
    
    if(err != cudaSuccess){
        fprintf(stderr, "Failed to allocate device vector B(error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    printf("Copy input data from the host memory to the cuda Device.");
    
    printf("\n********** lets copy the data **********\n");

    err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice); //serially coping data to GPU memory
    if(err != cudaSuccess){
        fprintf(stderr, "Failed to copy vector A from host to device(error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    if(err != cudaSuccess){
        fprintf(stderr, "Failed to copy vector A from host to device(error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // int threadsPerBlock = n < max_threads_per_block ? n :max_threads_per_block;
    // int blocksPerGrid = (n+threadsPerBlock-1)/threadsPerBlock;

    dim3 block(32, 32);
    dim3 grid(ceil(n/32), ceil(n/32)) ;

    //printf("CUDA kernal launch with %d threads of %d blocks\n", block[0],block[1]);

    vectorAdd<<< grid ,block >>>( d_A, d_B, d_C, n );

    err = cudaGetLastError(); //to findout the error in last kernal function call
    if(err != cudaSuccess){
        fprintf(stderr, "Failed to run vector addtion kernal B(error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    err = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    if(err != cudaSuccess){
        fprintf(stderr, "Failed to copy vector A from Device to Host(error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
}
int main(){
    unsigned int n = ARR_SIZE;

    float* h_A = (float*)malloc(sizeof(float) * n);
    float* h_B = (float*)malloc(sizeof(float) * n);
    float* h_C = (float*)malloc(sizeof(float) * n); //to save result
    
    for(int i=0;i<n; i++){
        for(int j=0; j<n; j++){
            h_A[i*n + j] = 2;
            h_B[i*n + j] = 2;
        }
    }
    vecAdd(h_A, h_B, h_C, n);

    //device function (CUDA Kernel) called from host does not have return type
    //CUDA runtime functions (execute in host side) can have return type
    int count=0;
    for(int i=0; i< n; i++){
        for(int j=0; j<n;j++)
            if(h_C[i] != 4.0f){
                printf("idx[%d] Value = %f\n", i, h_C[i]);
            }
            count+=1;
    }
    return(0);
}