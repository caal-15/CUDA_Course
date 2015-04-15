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

int main(int argc, char **argv){
  if (argc < 2){
    cout << "Usage: ./convolution max_size" << endl;
    return 0;
  }
  const int max_size = atoi(argv[1]);
  srand (time (NULL));

  ofstream x ("x.mio"),
  y_seq ("y_seq.mio"),
  y_con_g("y_con_g.mio"),
  y_con_c("y_con_c.mio"),
  y_con_t("y_con_t.mio");

  clock_t begin, end;
  double secs;
  float t_vec[32], t_mask[3], t_ans[32];
  fill_vector_random (t_vec, 32);
  fill_vector_random (t_mask, 3);
  convolution_kernel_con(t_vec, t_mask, t_ans, 32, 3);

  for (int i = 32; i <= max_size; i += 32){
    float vec[i], mask[9], c[i], d[i];
    fill_vector_random(vec, i);
    fill_vector_random(mask, 9);

    x << i << endl;

    begin = clock();
    convolution_kernel_seq(vec, mask, c, i, 9);
    end = clock();
    secs = double(end - begin) / CLOCKS_PER_SEC;
    y_seq << secs <<endl;

    begin = clock();
    convolution_kernel_con(vec, mask, d, i, 9);
    end = clock();
    secs = double(end - begin) / CLOCKS_PER_SEC;
    y_con_g << secs <<endl;

    if (!cmp_vector(c, d, i))
      cout << "SWW" << endl;

    begin = clock();
    convolution_kernel_constant(vec, mask, d, i, 9);
    end = clock();
    secs = double(end - begin) / CLOCKS_PER_SEC;
    y_con_c << secs <<endl;

    if (!cmp_vector(c, d, i))
      cout << "SWW" << endl;

    begin = clock();
    convolution_kernel_tiled(vec, mask, d, i, 9);
    end = clock();
    secs = double(end - begin) / CLOCKS_PER_SEC;
    y_con_t << secs <<endl;

    if (!cmp_vector(c, d, i))
      cout << "SWW" << endl;
  }


  return 0;
}
