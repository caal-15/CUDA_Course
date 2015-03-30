#include <cuda.h>
#include <bits/stdc++.h>

#define BLOCK_SIZE 32
#define EPSILON 0.1
#define MAX_MASK_SIZE 9

using namespace std;

__constant__ float g_mask[MAX_MASK_SIZE];

bool cmp_float (float a, float b){
  if (fabs (a - b) > EPSILON)
    return false;
  else
    return true;
}

void fill_vector_random (float *vec, int size, float max_size = 10.0){
  for (int i = 0; i < size; i++){
    vec[i] = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/max_size));
  }
}

bool cmp_vector(float *a, float *b, int size){
  for (int i = 0; i < size ; i++){
    if (!cmp_float (a[i], b[i]))
      return false;
  }
  return true;
}

void convolution_kernel_seq (float *vec, float *mask, float *ans, int vec_size, int mask_size){
  int start = 0;
  float con_val = 0.0;
  for (int i = 0; i < vec_size; i++){
    for (int j = 0; j < mask_size; j++){
      start = i - (mask_size / 2);
      if (start + j >= 0 && start + j < vec_size)
        con_val += vec[start + j] * mask[j];
    }
    ans[i] = con_val;
    con_val = 0.0;
  }
}

__global__ void convolution_kernel_kernel (float *vec, float *mask, float *ans, int vec_size, int mask_size){
  int pos = blockIdx.x * blockDim.x + threadIdx.x;
  float con_val = 0.0;
  int start = pos - (mask_size / 2);
  for (int i = 0; i < mask_size; i ++){
    if (start + i >= 0 && start + i < vec_size)
      con_val += vec[start + i] * mask[i];
  }
   ans[pos] = con_val;
}

void convolution_kernel_con (float *vec, float *mask, float *ans, int vec_size, int mask_size){
  float d_vec[vec_size], d_mask[mask_size], d_ans[vec_size];

  cudaMemcpy (d_vec, vec, vec_size * sizeof (float), cudaMemcpyHostToDevice);
  cudaMemcpy (d_mask, mask, mask_size * sizeof (float), cudaMemcpyHostToDevice);

  dim3 dimGrid (ceil (vec_size / float (BLOCK_SIZE)), 1, 1);
  dim3 dimBlock (BLOCK_SIZE, 1, 1);

  convolution_kernel_kernel<<<dimGrid, dimBlock>>> (d_vec, d_mask, d_ans, vec_size, mask_size);
  cudaDeviceSynchronize();

  cudaMemcpy (ans, d_ans, vec_size * sizeof (float), cudaMemcpyDeviceToHost);
  cudaFree (d_vec);
  cudaFree (d_mask);
  cudaFree (d_ans);
}

__global__ void convolution_kernel_kernel_constant (float *vec, float *ans, int vec_size, int mask_size){
  int pos = blockIdx.x * blockDim.x + threadIdx.x;
  float con_val = 0.0;
  int start = pos - (mask_size / 2);
  for (int i = 0; i < mask_size; i ++){
    if (start + i >= 0 && start + i < vec_size)
      con_val += vec[start + i] * g_mask[i];
  }
   ans[pos] = con_val;
}

void convolution_kernel_constant(float *vec, float *mask, float *ans, int vec_size, int mask_size){
  float d_vec[vec_size], d_ans[vec_size];

  cudaMemcpy (d_vec, vec, vec_size * sizeof (float), cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol (g_mask, mask, mask_size * sizeof (float));

  dim3 dimGrid (ceil (vec_size / float (BLOCK_SIZE)), 1, 1);
  dim3 dimBlock (BLOCK_SIZE, 1, 1);
  convolution_kernel_kernel_constant<<<dimGrid, dimBlock>>> (d_vec, d_ans, vec_size, mask_size);

  cudaMemcpy (ans, d_ans, vec_size * sizeof (float), cudaMemcpyDeviceToHost);
  cudaFree (d_vec);
  cudaFree (d_ans);
}

__global__ void convolution_kernel_kernel_tiled (float *vec, float *ans, int vec_size, int mask_size){
  int pos = blockIdx.x * blockDim.x + threadIdx.x;
  int last_pos = (blockIdx.x - 1) * blockDim.x + threadIdx.x;
  int offset = mask_size / 2;
  __shared__ float s_vec[BLOCK_SIZE + offset * 2];




}

int main(){
  return 0;
}
