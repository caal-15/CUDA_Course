#include <cuda.h>
#include <bits/stdc++.h>

#define BLOCK_SIZE 32
#define EPSILON 0.1
#define MAX_MASK_SIZE 15

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

void print_vector (float *a, int size){
  for (int i = 0; i < size; i++)
    cout << a[i] << endl;
  cout << "--------------" << endl;
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
  float *d_vec;
  float *d_mask;
  float *d_ans;

  cudaMalloc(&d_vec, vec_size * sizeof(float));
  cudaMalloc(&d_mask, mask_size * sizeof(float));
  cudaMalloc(&d_ans, vec_size * sizeof(float));

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
  float *d_vec;
  float *d_ans;

  cudaMalloc(&d_vec, vec_size * sizeof(float));
  cudaMalloc(&d_ans, vec_size * sizeof(float));

  cudaMemcpy (d_vec, vec, vec_size * sizeof (float), cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol (g_mask, mask, mask_size * sizeof (float));

  dim3 dimGrid (ceil (vec_size / float (BLOCK_SIZE)), 1, 1);
  dim3 dimBlock (BLOCK_SIZE, 1, 1);
  convolution_kernel_kernel_constant<<<dimGrid, dimBlock>>> (d_vec, d_ans, vec_size, mask_size);

  cudaMemcpy (ans, d_ans, vec_size * sizeof (float), cudaMemcpyDeviceToHost);
  cudaFree (d_vec);
  cudaFree (d_ans);
}

__global__ void convolution_kernel_kernel_tiled (float *vec, float *ans, int vec_size, const int mask_size){
  int pos = blockIdx.x * blockDim.x + threadIdx.x;
  int last_pos = (blockIdx.x - 1) * blockDim.x + threadIdx.x;
  int next_pos = (blockIdx.x + 1) * blockDim.x + threadIdx.x;
  int offset = mask_size / 2;
  float sum;
  __shared__ float s_vec[BLOCK_SIZE + MAX_MASK_SIZE -1];

  if (threadIdx.x >= blockDim.x - offset){
    if (last_pos < 0)
      s_vec[threadIdx.x - (blockDim.x - offset)] = 0;
    else
      s_vec[threadIdx.x - (blockDim.x - offset)] = vec[last_pos];
  }

  s_vec[threadIdx.x + offset] = vec[pos];

  if (threadIdx.x < offset){
    if (next_pos >= vec_size)
      s_vec[threadIdx.x + blockDim.x + offset] = 0;
    else
      s_vec[threadIdx.x + blockDim.x + offset] = vec[next_pos];
  }

  __syncthreads();

  for (int i = 0; i < mask_size; i++)
    sum += g_mask[i] * s_vec[threadIdx.x + i];

  ans[pos] = sum;

}

void convolution_kernel_tiled(float *vec, float *mask, float *ans, int vec_size, int mask_size){
  float *d_vec;
  float *d_ans;

  cudaMalloc(&d_vec, vec_size * sizeof(float));
  cudaMalloc(&d_ans, vec_size * sizeof(float));

  cudaMemcpy (d_vec, vec, vec_size * sizeof (float), cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol (g_mask, mask, mask_size * sizeof (float));

  dim3 dimGrid (ceil (vec_size / float (BLOCK_SIZE)), 1, 1);
  dim3 dimBlock (BLOCK_SIZE, 1, 1);
  convolution_kernel_kernel_tiled<<<dimGrid, dimBlock>>> (d_vec, d_ans, vec_size, mask_size);

  cudaMemcpy (ans, d_ans, vec_size * sizeof (float), cudaMemcpyDeviceToHost);
  cudaFree (d_vec);
  cudaFree (d_ans);
}

int main(){
  srand (time (NULL));
  float vec[1024], mask[9], c[1024], d[1024], e[1024], f[1024];
  fill_vector_random(vec, 1024);
  fill_vector_random(mask, 9);

  convolution_kernel_seq(vec, mask, c, 1024, 9);
  convolution_kernel_con(vec, mask, d, 1024, 9);

  if (!cmp_vector(c, d, 1024))
    cout << "SWW" << endl;

  convolution_kernel_constant(vec, mask, e, 1024, 9);

  if (!cmp_vector(c, e, 1024))
    cout << "SWW" << endl;

  convolution_kernel_tiled(vec, mask, f, 1024, 9);

  if (!cmp_vector(c, f, 1024))
    cout << "SWW" << endl;

  print_vector(c, 5);
  print_vector(d, 5);
  print_vector(e, 5);
  print_vector(f, 5);

  return 0;
}
