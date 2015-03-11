#include <cuda.h>
#include <bits/stdc++.h>

#define BLOCK_SIZE 32
#define EPSILON 0.1

using namespace std;

bool cmp_float(float a, float b){
  if (fabs(a - b) > EPSILON)
    return false;
  else
    return true;
}

void fill_matrix_random(float *mat, int rows, int cols){
  for (int i = 0; i < rows; i++){
    for (int j = 0; j < cols; j++){
      mat[i * cols + j] = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/10.0));;
    }
  }
}

bool check_matrix(float *A, float *B, int rows, int cols){

  for (int i = 0; i < rows; i++){
    for (int j = 0; j < cols; j++){
      //cout << A[i * cols + j] << " the fuck?? "  << B[i * cols +j] << endl;
      if (!cmp_float(A[i * cols + j], B[i * cols +j]))
        return false;
    }
  }
  return true;
}

void print_matrix(float *mat, int rows, int cols){
  cout << "------------" << endl;
  for (int i = 0; i < rows; i++){
    for (int j = 0; j < cols; j++){
      cout << mat[i * cols + j] << " ";
    }
    cout << endl;
  }
  cout << "------------" << endl;
}

void mat_mul_seq(float *m_A, float *m_B, float *m_C, int A_rows, int A_cols, int B_rows, int B_cols){
  float sum;
  for(int i = 0; i < A_rows; i++){
    for (int j = 0; j < B_cols; j++){
      sum = 0.0;
      for (int k = 0; k < A_cols; k++){
        sum += m_A[i * A_cols + k] * m_B[k * B_cols + j];
      }
      m_C[i * B_cols + j] = sum;
    }
  }
}
__global__ void mat_mul_kernel(float *m_A, float *m_B, float *m_C, int A_rows, int A_cols, int B_rows, int B_cols){
  float sum = 0;
  int row = threadIdx.y + blockIdx.y * blockDim.y;
  int col = threadIdx.x + blockIdx.x * blockDim.x;
  if(row < A_rows && col < B_cols){
    for(int i = 0; i < A_cols; i ++){
      sum += m_A[row * A_cols + i] * m_B[i * B_cols + col];
    }
    m_C[row * B_cols + col] = sum;
  }
}

__global__ void mat_mul_kernel_tiled(float *m_A, float *m_B, float *m_C, int A_rows, int A_cols, int B_rows, int B_cols){
  __shared__ float s_A[BLOCK_SIZE][BLOCK_SIZE], s_B[BLOCK_SIZE][BLOCK_SIZE];

  int blockRow = blockIdx.y;
  int blockCol = blockIdx.x;

  int threadRow = threadIdx.y;
  int threadCol = threadIdx.x;

  int row = blockRow * BLOCK_SIZE + threadRow;
  int col = blockCol * BLOCK_SIZE + threadCol;

  float sum = 0;
  for (int sm = 0; sm < ceil (A_cols / float (BLOCK_SIZE)); sm++){
    if (row < A_rows && (sm * BLOCK_SIZE + threadCol) < A_cols){
      s_A[threadRow][threadCol] = m_A[(row) * A_cols + (sm * BLOCK_SIZE + threadCol)];
    } else{
      s_A[threadRow][threadCol] = 0.0;
    }
    if (col < B_cols && (threadRow + sm * BLOCK_SIZE) < B_rows){
      s_B[threadRow][threadCol] = m_B[(threadRow + sm * BLOCK_SIZE) * B_cols + (col)];
    } else{
      s_B[threadRow][threadCol] = 0.0;
    }
    __syncthreads();
    for (int i = 0; i < BLOCK_SIZE; i++){
      sum += s_A[threadRow][i] * s_B[i][threadCol];
    }
    __syncthreads();
    if (row < A_rows && col < B_cols)
      m_C[row * B_cols + col] = sum;
  }

}

void mat_mul_con(float *m_A, float *m_B, float *m_C, int A_rows, int A_cols, int B_rows, int B_cols){
    int A_size = A_rows * A_cols * sizeof(float);
    int B_size = B_rows * B_cols * sizeof(float);
    int C_size = A_rows * B_cols * sizeof(float);

    float *d_A, *d_B, *d_C;
    //1. Allocate memory for d_A, etc. on the device (cudaMalloc)
    cudaMalloc(&d_A, A_size);
    cudaMalloc(&d_B, B_size);
    cudaMalloc(&d_C, C_size);
    //2. Copy Data from host to d_A, etc. (cudaMemcpy)
    cudaMemcpy(d_A, m_A, A_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, m_B, B_size, cudaMemcpyHostToDevice);
    //3. Kernel Launch Code
    dim3 dimGrid(ceil(max(A_cols, B_cols) / float (BLOCK_SIZE)), ceil(max(A_rows, B_rows) / float (BLOCK_SIZE)), 1);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
    mat_mul_kernel<<<dimGrid, dimBlock>>> (d_A, d_B, d_C, A_rows, A_cols, B_rows, B_cols);
    cudaDeviceSynchronize();
    //4. Copy d_C to C from device, free device memory (cusdaFree), sync if neccessary
    cudaMemcpy (m_C, d_C, C_size, cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

void mat_mul_con_tiled(float *m_A, float *m_B, float *m_C, int A_rows, int A_cols, int B_rows, int B_cols){
    int A_size = A_rows * A_cols * sizeof(float);
    int B_size = B_rows * B_cols * sizeof(float);
    int C_size = A_rows * B_cols * sizeof(float);

    float *d_A, *d_B, *d_C;
    //1. Allocate memory for d_A, etc. on the device (cudaMalloc)
    cudaMalloc(&d_A, A_size);
    cudaMalloc(&d_B, B_size);
    cudaMalloc(&d_C, C_size);
    //2. Copy Data from host to d_A, etc. (cudaMemcpy)
    cudaMemcpy(d_A, m_A, A_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, m_B, B_size, cudaMemcpyHostToDevice);
    //3. Kernel Launch Code
    dim3 dimGrid(ceil(max(A_cols, B_cols) / float (BLOCK_SIZE)), ceil(max(A_rows, B_rows) / float (BLOCK_SIZE)), 1);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
    mat_mul_kernel_tiled<<<dimGrid, dimBlock>>> (d_A, d_B, d_C, A_rows, A_cols, B_rows, B_cols);
    cudaDeviceSynchronize();
    //4. Copy d_C to C from device, free device memory (cusdaFree), sync if neccessary
    cudaMemcpy (m_C, d_C, C_size, cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main(int argc, char **argv){
  if (argc < 5){
    cout << "Usage: ./mul max_number step offset_A, offset_B" << endl;
    return 0;
  }
  const int max_number = atoi(argv[1]),
  step = atoi(argv[2]),
  offset_A = atoi(argv[3]),
  offset_B = atoi(argv[4]);
  srand (time(NULL));

  ofstream x("x.mio"),
  y_seq("y_seq.mio"),
  y_con("y_con.mio"),
  y_con_tiled("y_con_tiled.mio");

  clock_t begin, end;
  double elapsed_secs;

  for (int i = step; i <= max_number; i += step){
    float *A, *B, *C, *D;
    A = (float*) malloc((i + offset_A) * i * sizeof(float));
    B = (float*) malloc((i + offset_B) * i * sizeof(float));
    C = (float*) malloc((i + offset_A) * (i + offset_B) * sizeof(float));
    D = (float*) malloc((i + offset_A) * (i + offset_B) * sizeof(float));

    x << i << endl;

    fill_matrix_random(A, i + offset_A, i);
    fill_matrix_random(B, i, i + offset_B);

    begin = clock();
    mat_mul_seq(A, B, C, i + offset_A, i, i, i + offset_B);
    end = clock();
    elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    y_seq << elapsed_secs << endl;

    begin = clock();
    mat_mul_con(A, B, D, i + offset_A, i, i, i + offset_B);
    end = clock();
    elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    y_con << elapsed_secs << endl;

    cout << "-----------" << endl << "Not Tiled: ";
    if (check_matrix(C, D, i + offset_A, i + offset_B))
      cout << "All good" << endl;
    else
      cout << "Something Went Wrong" << endl;
    cout << "-----------" << endl;

    

    begin = clock();
    mat_mul_con_tiled(A, B, D, i + offset_A, i, i, i + offset_B);
    end = clock();
    elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    y_con_tiled << elapsed_secs << endl;

    cout << "Tiled: ";
    if (check_matrix(C, D, i + offset_A, i + offset_B))
      cout << "All good" << endl;
    else
      cout << "Something Went Wrong" << endl;
    cout << "-----------" << endl;

    free(A);
    free(B);
    free(C);
    free(D);
  }

  return 0;
}
