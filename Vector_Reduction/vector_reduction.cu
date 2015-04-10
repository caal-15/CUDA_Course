#include <cuda.h>
#include <bits/stdc++.h>

#define BLOCK_SIZE 32

using namespace std;

void fill_vector_random (int *vec, int size, int max = 10){
  for (int i = 0; i < size; i++)
    vec[i] = 1;
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
  while (size >= BLOCK_SIZE * 2){
    
    int *d_vec, *d_out;
    cudaMalloc (&d_vec, size * sizeof(int));
    cudaMalloc (&d_out, (size / (BLOCK_SIZE * 2)) * sizeof(int));

    cudaMemcpy (d_vec, vec, size * sizeof(int), cudaMemcpyHostToDevice);
    dim3 dimGrid (ceil (size / float(BLOCK_SIZE)), 1, 1);
    dim3 dimBlock (BLOCK_SIZE, 1, 1);

    vector_reduction_kernel<<<dimGrid, dimBlock>>> (d_vec, d_out);
    cudaDeviceSynchronize();
    size = size / (BLOCK_SIZE * 2);

    cudaMemcpy (vec, d_out, size * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree (d_vec);
    cudaFree (d_out);

  }

  return vector_reduction_seq(vec, size);

}

int main(int argc, char **argv){
  if (argc < 2){
    cout << "Usage: ./reduction max_vector_size" << endl;
    return 0;
  }
  const int max_size = atoi(argv[1]);
  srand (time (NULL));
  ofstream x("x.mio"),
  y_seq ("y_seq.mio"),
  y_con ("y_con.mio");
  clock_t begin, end;
  double secs;
  int ans1, ans2;
  int first_vec[64];
  fill_vector_random(first_vec, 64);
  vector_reduction_con (first_vec, 64);
  for (int i = 64; i <= max_size; i += 64){
    int vec[i];
    fill_vector_random (vec, i);
    x << i << endl;

    begin = clock();
    ans1 = vector_reduction_seq (vec, i);
    end = clock();
    secs = double(end - begin) / CLOCKS_PER_SEC;
    y_seq << secs << endl;

    begin = clock();
    ans2 = vector_reduction_con (vec, i);
    end = clock();
    secs = double(end - begin) / CLOCKS_PER_SEC;
    y_con << secs << endl;

    if (ans1 != ans2)
      cout << "SWW" << endl << ans1 << " " << ans2 << endl;
  }

  return 0;
}
