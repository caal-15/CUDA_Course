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



__global__ void img2gray_kernel(unsigned char *data, unsigned char *result, int width, int height){
  int row = blockIdx.y*blockDim.y+threadIdx.y;
  int col = blockIdx.x*blockDim.x+threadIdx.x;
  if((row < height) && (col < width)){
    result[row*width+col] = data[(row*width+col)*3+2]*0.299 + data[(row*width+col)*3+1]*0.587 \
    + data[(row*width+col)*3]*0.114;
  }
}

void img2gray(unsigned char *data, unsigned char *result, int width, int height){
  unsigned char *d_data, *d_result;

  cudaMalloc (&d_data, sizeof(unsigned char) * width * height * 3);
  cudaMalloc (&d_result, sizeof(unsigned char) * width * height);

  cudaMemcpy(d_data, data, sizeof(unsigned char) * width * height * 3, cudaMemcpyHostToDevice);

  dim3 dimGrid (ceil (width / float (BLOCK_SIZE)), ceil (height / float (BLOCK_SIZE)), 1);
  dim3 dimBlock (BLOCK_SIZE, BLOCK_SIZE, 1);

  img2gray_kernel <<<dimGrid, dimBlock>>> (d_data, d_result, width, height);
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );
  cudaMemcpy(result, d_result, sizeof(unsigned char) * width * height, cudaMemcpyDeviceToHost);

  cudaFree(d_data);
  cudaFree(d_result);
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

int main (int argc, char **argv){
  if (argc < 2){
    cout << "Usage: ./concolution2d image_name" << endl;
  }

  char mask[] = {-1,0,1,-2,0,2,-1,0,1};
  gpuErrchk(cudaMemcpyToSymbol (SOBEL, mask, 3 * 3 * sizeof (char)));

  Mat image, result;
  image = imread("../img1.jpg",0);

  Size s = image.size();
  int Row = s.width;
  int Col = s.height;

  unsigned char *Gray = (unsigned char*)malloc (sizeof (unsigned char)* Row * Col);
  unsigned char *Out = (unsigned char*)malloc (sizeof (unsigned char)* Row * Col);

  Gray = image.data;

  convolution_con_tiled(Gray, Out, Row, Col);
  result.create(Row, Col, CV_8UC1);
  result.data = Out;
  imwrite ("miau.jpg", result);




  return 0;
}
