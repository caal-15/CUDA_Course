#include <cuda.h>
#include <bits/stdc++.h>
#include "opencv2/opencv.hpp"

#define BLOCK_SIZE 32
#define EPSILON 0.1
#define MASK_SIZE 3
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

using namespace std;
using namespace cv;

__constant__ char SOBEL[MASK_SIZE * MASK_SIZE];


inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__device__ unsigned char clamp(int value){
  if(value < 0)
    value = 0;
  else
    if(value > 255)
      value = 255;
  return (unsigned char)value;
}

__global__ void convolution_kernel_tiled (unsigned char *data, unsigned char *result, int width, int height) {
   __shared__ int s_data[BLOCK_SIZE + MASK_SIZE - 1][BLOCK_SIZE + MASK_SIZE - 1];
  // First batch loading
  const int radius = MASK_SIZE / 2;
  int dest = threadIdx.y * BLOCK_SIZE + threadIdx.x,
     destY = dest / (BLOCK_SIZE + MASK_SIZE - 1), destX = dest % (BLOCK_SIZE + MASK_SIZE - 1),
     srcY = blockIdx.y * BLOCK_SIZE + destY - radius,
     srcX = blockIdx.x * BLOCK_SIZE + destX - radius,
     src = srcY * width + srcX;
  if (srcY >= 0 && srcY < height && srcX >= 0 && srcX < width)
     s_data[destY][destX] = data[src];
  else
     s_data[destY][destX] = 0;

  // Second batch loading
  dest = threadIdx.y * BLOCK_SIZE + threadIdx.x + BLOCK_SIZE * BLOCK_SIZE;
  destY = dest / (BLOCK_SIZE + MASK_SIZE - 1), destX = dest % (BLOCK_SIZE + MASK_SIZE - 1);
  srcY = blockIdx.y * BLOCK_SIZE + destY - radius;
  srcX = blockIdx.x * BLOCK_SIZE + destX - radius;
  src = srcY * width + srcX;
  if (destY < BLOCK_SIZE + MASK_SIZE - 1) {
     if (srcY >= 0 && srcY < height && srcX >= 0 && srcX < width)
        s_data[destY][destX] = data[src];
     else
        s_data[destY][destX] = 0;
  }

  __syncthreads();

  int sum = 0;
  int y, x;
  for (y = 0; y < MASK_SIZE; y++)
     for (x = 0; x < MASK_SIZE; x++)
        sum += s_data[threadIdx.y + y][threadIdx.x + x] * SOBEL[y * MASK_SIZE + x];
  y = blockIdx.y * BLOCK_SIZE + threadIdx.y;
  x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  if (y < height && x < width)
    result[y * width + x] = clamp(sum);
  __syncthreads();
}

void convolution_con_tiled (unsigned char *data, unsigned char *result, int width, int height){
  unsigned char *d_data, *d_result;

  cudaMalloc (&d_data, sizeof(unsigned char) * width * height);
  cudaMalloc (&d_result, sizeof(unsigned char) * width * height);

  cudaMemcpy (d_data, data, sizeof(unsigned char) * width * height, cudaMemcpyHostToDevice);

  dim3 dimGrid (ceil (width / float (BLOCK_SIZE)), ceil (height / float (BLOCK_SIZE)), 1);
  dim3 dimBlock (BLOCK_SIZE, BLOCK_SIZE, 1);

  convolution_kernel_tiled<<<dimGrid, dimBlock>>> (d_data, d_result, width, height);
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );

  cudaMemcpy(result, d_result, sizeof(unsigned char) * width * height, cudaMemcpyDeviceToHost);

  cudaFree(d_data);
  cudaFree(d_result);
}

__global__ void convolution_kernel_global (unsigned char *data, char *mask, unsigned char *result, int mask_size, int width, int height){
  unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
  int sum = 0;
  int start_row = row - (mask_size / 2);
  int start_col = col - (mask_size / 2);
  for(int i = 0; i < mask_size; i++){
    for(int j = 0; j < mask_size; j++ ){
      if((start_col + j >= 0 && start_col + j < width) && (start_row + i >= 0 && start_row + i < height)){
        sum += data[(start_row + i) * width + (start_col + j)] * mask[i * mask_size + j];
      }
    }
  }
  result[row * width + col] = clamp(sum);
}

void convolution_con_global (unsigned char *data, char *mask, unsigned char *result, int mask_size, int width, int height){
  unsigned char *d_data, *d_result;
  char *d_mask;

  cudaMalloc (&d_data, sizeof(unsigned char) * width * height);
  cudaMalloc (&d_mask, sizeof(char) * mask_size * mask_size);
  cudaMalloc (&d_result, sizeof(unsigned char) * width * height);

  cudaMemcpy (d_data, data, sizeof(unsigned char) * width * height, cudaMemcpyHostToDevice);
  cudaMemcpy (d_mask, mask, sizeof(char) * mask_size * mask_size, cudaMemcpyHostToDevice);

  dim3 dimGrid (ceil (width / float (BLOCK_SIZE)), ceil (height / float (BLOCK_SIZE)), 1);
  dim3 dimBlock (BLOCK_SIZE, BLOCK_SIZE, 1);

  convolution_kernel_global<<<dimGrid, dimBlock>>> (d_data, d_mask, d_result, mask_size, width, height);
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );

  cudaMemcpy(result, d_result, sizeof(unsigned char) * width * height, cudaMemcpyDeviceToHost);

  cudaFree(d_data);
  cudaFree(d_mask);
  cudaFree(d_result);
}

__global__ void convolution_kernel_constant (unsigned char *data, unsigned char *result, int width, int height){
  unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
  int sum = 0;
  int start_row = row - (MASK_SIZE / 2);
  int start_col = col - (MASK_SIZE / 2);
  for(int i = 0; i < MASK_SIZE; i++){
    for(int j = 0; j < MASK_SIZE; j++ ){
      if((start_col + j >= 0 && start_col + j < width) && (start_row + i >= 0 && start_row + i < height)){
        sum += data[(start_row + i) * width + (start_col + j)] * SOBEL[i * MASK_SIZE + j];
      }
    }
  }
  result[row * width + col] = clamp(sum);
}

void convolution_con_constant (unsigned char *data, unsigned char *result, int width, int height){
  unsigned char *d_data, *d_result;

  cudaMalloc (&d_data, sizeof(unsigned char) * width * height);
  cudaMalloc (&d_result, sizeof(unsigned char) * width * height);

  cudaMemcpy (d_data, data, sizeof(unsigned char) * width * height, cudaMemcpyHostToDevice);

  dim3 dimGrid (ceil (width / float (BLOCK_SIZE)), ceil (height / float (BLOCK_SIZE)), 1);
  dim3 dimBlock (BLOCK_SIZE, BLOCK_SIZE, 1);

  convolution_kernel_constant<<<dimGrid, dimBlock>>> (d_data, d_result, width, height);
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );

  cudaMemcpy(result, d_result, sizeof(unsigned char) * width * height, cudaMemcpyDeviceToHost);

  cudaFree(d_data);
  cudaFree(d_result);
}

int main (int argc, char **argv){

  char mask[] = {-1,0,1,-2,0,2,-1,0,1};
  gpuErrchk(cudaMemcpyToSymbol (SOBEL, mask, MASK_SIZE * MASK_SIZE * sizeof (char)));

  Mat image, result, result_seq;
  clock_t begin, end;
  stringstream path, outpath;
  ofstream x ("x.mio"),
  y_seq ("y_seq.mio"),
  y_con_g ("y_con_g.mio"),
  y_con_c ("y_con_c.mio"),
  y_con_t ("y_con_t.mio");

  for (int i = 1; i < 7; i++){
    path << "../images/img"  << i << ".jpg";
    image = imread(path.str(), 0);
    path.str("");

    Size s = image.size();
    int width = s.width;
    int height = s.height;

    x << width * height << endl;

    unsigned char *gray = (unsigned char*)malloc (sizeof (unsigned char)* width * height);
    unsigned char *out_global = (unsigned char*)malloc (sizeof (unsigned char)* width * height);
    unsigned char *out_constant = (unsigned char*)malloc (sizeof (unsigned char)* width * height);
    unsigned char *out_tiled = (unsigned char*)malloc (sizeof (unsigned char)* width * height);

    gray = image.data;
    float seq_sum = 0,
    global_sum = 0,
    constant_sum = 0,
    tiled_sum = 0;

    for (int j = 0; j < 20; j++){

      begin = clock();
      Sobel (image, result_seq, CV_8UC1, 1, 0, 3, 1, 0, BORDER_DEFAULT);
      end = clock();
      seq_sum += float (end - begin) / CLOCKS_PER_SEC;

      begin = clock();
      convolution_con_global(gray, mask, out_global, 3, width, height);
      end = clock();
      global_sum += float (end - begin) / CLOCKS_PER_SEC;

      begin = clock();
      convolution_con_constant(gray, out_constant, width, height);
      end = clock();
      constant_sum += float (end - begin) / CLOCKS_PER_SEC;

      begin = clock();
      convolution_con_tiled(gray, out_tiled, width, height);
      end = clock();
      tiled_sum += float (end - begin) / CLOCKS_PER_SEC;
    }

    seq_sum /= 20.0;
    y_seq << seq_sum << endl;
    global_sum /= 20.0;
    y_con_g << global_sum << endl;
    constant_sum /= 20.0;
    y_con_c << constant_sum << endl;
    tiled_sum /= 20.0;
    y_con_t << tiled_sum << endl;



    outpath << "../outputs/" << i << "_miau_seq" << ".jpg";
    imwrite (outpath.str(), result_seq);
    outpath.str("");

    result.create(height, width, CV_8UC1);
    result.data = out_global;
    outpath << "../outputs/" << i << "_miau_global" << ".jpg";
    imwrite (outpath.str(), result);
    outpath.str("");

    result.data = out_constant;
    outpath << "../outputs/" << i << "_miau_constant" << ".jpg";
    imwrite (outpath.str(), result);
    outpath.str("");

    result.data = out_tiled;
    outpath << "../outputs/" << i << "_miau_tiled" << ".jpg";
    imwrite (outpath.str(), result);
    outpath.str("");

    image.release();
    result.release();
    result_seq.release();
    free (out_global);
    free (out_constant);
    free (out_tiled);

  }



  return 0;
}
