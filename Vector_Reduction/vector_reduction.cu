#include <cuda.h>
#include <bits/stdc++.h>

#define BLOCK_SIZE 32

using namespace std;

void fill_vector_random (int *vec, int size, int max = 10){
  for (int i = 0; i < size; i++)
    vec[i] = rand() % max;
}

void print_vector (int *vec, int size){
  for (int i = 0; i < size; i++)
    cout << vec[i] << endl;
  cout << "___________" << endl;
}

int vector_reduction_seq (int *vec, int size){
  int ans = 0;
  for (int i = 0; i < size; i++){
    ans += vec[i];
  }
  return ans;
}

__global__ void vector_reduction_kernel (int *vec, int *out){
  int pos = blockIdx.x * blockDim.x * 2 + threadIdx.x;

  __shared__ int svec[BLOCK_SIZE];
  svec[threadIdx.x] = vec[pos] + vec[pos + blockDim.x];
  __syncthreads();

  for (int i = blockDim.x / 2; i > 0; i >>= 1){
    if (threadIdx.x < i)
      svec[threadIdx.x] += svec[threadIdx.x + i];
    __syncthreads();
  }

  if (threadIdx.x == 0)
    out[blockIdx.x] = svec[0];
}

int vector_reduction_con (int *vec, int size){
  while (size > BLOCK_SIZE * 2){
    int *d_vec, *d_out;
    cudaMalloc (&d_vec, size * sizeof(int));
    cudaMalloc (&d_out, (size / (BLOCK_SIZE * 2)) * sizeof(int));

    cudaMemcpy (d_vec, vec, size * sizeof(int), cudaMemcpyHostToDevice);
    dim3 dimGrid (ceil (size / float(BLOCK_SIZE)), 1, 1);
    dim3 dimBlock (BLOCK_SIZE, 1, 1);

    vector_reduction_kernel<<<dimGrid, dimBlock>>> (d_vec, d_out);
    size = size / (BLOCK_SIZE * 2);

    cudaMemcpy (vec, d_out, size * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree (d_vec);
    cudaFree (d_out);

  }

  return vector_reduction_seq(vec, size);

}

int main(){
  int size = 1024;
  srand (time (NULL));
  int vec[size];
  fill_vector_random(vec, size);

  cout << vector_reduction_seq(vec, size) << endl;
  cout << vector_reduction_con(vec, size) << endl;

  return 0;
}
