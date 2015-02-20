#include <cuda.h>
#include <bits/stdc++.h>

using namespace std;

void fill_matrix_random(int *mat, int size){
  srand (time(NULL));
  for (int i = 0; i < size; i++){
    for (int j = 0; j < size; j++){
      mat[i * size + j] = rand() % 256;
    }
  }
}

void print_matrix(int *mat, int size){
  cout << "------------" << endl;
  for (int i = 0; i < size; i++){
    for (int j = 0; j < size; j++){
      cout << mat[i * size + j] << " ";
    }
    cout << endl;
  }
  cout << "------------" << endl;
}

void mat_mul_seq(int *m_A, int *m_B, int *m_C, int size){
  int sum;
  for(int i = 0; i < size; i++){
    for (int j = 0; j < size; j++){
      sum = 0;
      for (int k = 0; k < size; k++){
        sum += m_A[i * size + k] * m_B[k * size + j];
      }
      m_C[i * size + j] = sum;
    }
  }
}
__global__ void mat_mul_kernel(int *m_A, int *m_B, int *m_C, int size){
  int sum = 0;
  int row = threadIdx.y + blockIdx.y * blockDim.y;
  int col = threadIdx.x + blockIdx.x * blockDim.x;  
  if(row < size && col < size){
    for(int i = 0; i < size; i ++){
      sum += m_A[row * size + i] * m_B[i * size + col];
    }
    m_C[row * size + col] = sum;
  }
}

void mat_mul_con(int *m_A, int *m_B, int *m_C, int size){
    int total_size = size * size * sizeof(int);
    
    int *d_A, *d_B, *d_C;
    //1. Allocate memory for d_A, etc. on the device (cudaMalloc)
    cudaMalloc(&d_A, total_size);
    cudaMalloc(&d_B, total_size);
    cudaMalloc(&d_C, total_size);
    //2. Copy Data from host to d_A, etc. (cudaMemcpy)
    cudaMemcpy(d_A, m_A, total_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, m_B, total_size, cudaMemcpyHostToDevice);
    //3. Kernel Launch Code
    dim3 dimGrid(ceil(size/32.0), ceil(size/32.0), 1);
    dim3 dimBlock(32, 32, 1);
    mat_mul_kernel<<<dimGrid, dimBlock>>> (d_A, d_B, d_C, size);
    //4. Copy d_C to C from device, free device memory (cusdaFree), sync if neccessary
    cudaMemcpy (m_C, d_C, total_size, cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main(int argc, char **argv){
  if (argc < 3){
    cout << "Usage: ./mul max_number step" << endl;
    return 0;
  }
  
  const int max_num = atoi(argv[1]);
  const int step = atoi(argv[2]);
  int *A, *B, *C;
  ofstream x("x.mio");
  ofstream y_seq("y_seq.mio");
  ofstream y_con("y_con.mio");
  for (int i = step; i <= max_num; i += step){
    //cout << "here " << i << endl;
    A = (int *)malloc(i * i * sizeof(int));
    B = (int *)malloc(i * i * sizeof(int));
    C = (int *)malloc(i * i * sizeof(int));
    fill_matrix_random(A, i);
    fill_matrix_random(B, i);
    x << i << endl;
    
    //Measure seq time
    clock_t begin = clock();
    mat_mul_seq(A, B, C, i);
    clock_t end = clock();
    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    y_seq << elapsed_secs << endl;
    
    //Measure concurrent time
    begin = clock();
    mat_mul_con(A, B, C, i);
    end = clock();
    elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    y_con << elapsed_secs << endl;
    
    free(A);
    free(B);
    free(C);
  }
  return 0;
}
