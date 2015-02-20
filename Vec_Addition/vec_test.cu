//Test 1
#include<cuda.h>
#include <bits/stdc++.h>

using namespace std;
//Sequential
//Vector addition kernel
void vecAdd(float *h_A, float *h_B, float *h_C, int n){
    int i;
    for (i = 0; i < n; i++)
        h_C[i] = h_A[i] + h_B[i];
}


//Parallel
__global__ void vecAddP (float *A, float *B, float *C, int n){
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if(i < n)
        C[i] = A[i] + B[i];
}

void vectorAdd(float *A, float *B, float *C, int n){
    int size = n * sizeof(float);
    
    float *d_A, *d_B, *d_C;
    //1. Allocate memory for d_A, etc. on the device (cudaMalloc)
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);
    //2. Copy Data from host to d_A, etc. (cudaMemcpy)
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
    //3. Kernel Launch Code
    vecAddP<<<ceil(n/256.0), 256>>> (d_A, d_B, d_C, n);
    //4. Copy d_C to C from device, free device memory (cusdaFree), sync if neccessary
    cudaMemcpy (C, d_C, size, cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

}
    
int main(){
    //Memalloc A, B, C
    //I/O to read A, B, C
    float A[5];
    float B[5];
    float C[5];
    float j = 0.0;
    int n = 5;
    for (int i = 0; i < 5; i = i + 1, j = j + 1.0){
        A[i] = j;
        B[i] = j +1.5;
    }
    vectorAdd(A, B, C, n);
  for (int i = 0 ; i < 5; i++){
      cout << C[i] << endl;
  }
}



