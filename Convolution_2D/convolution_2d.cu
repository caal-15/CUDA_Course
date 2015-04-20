#include <cuda.h>
#include <bits/stdc++.h>
#include "opencv2/opencv.hpp"

#define BLOCK_SIZE 32
#define EPSILON 0.1
#define MASK_SIZE 3

using namespace std;
using namespace cv;

__constant__ char SOBEL[MASK_SIZE * MASK_SIZE];


__device__ unsigned char clamp(int value){
  if(value < 0)
    value = 0;
  else
    if(value > 255)
      value = 255;
  return (unsigned char)value;
}

__global__ void convolution_kernel_tiled(unsigned char *data, unsigned char *result, int width, int height){
  // Data cache: threadIdx.x , threadIdx.y
  int n = MASK_SIZE / 2;
  __shared__ int s_data[BLOCK_SIZE + MASK_SIZE - 1][BLOCK_SIZE + MASK_SIZE - 1];

  // global mem address of this thread
  const int pos = threadIdx.x + (blockIdx.x * blockDim.x) + (threadIdx.y * width) + (blockIdx.y * blockDim.y) * width;

  // load cache (32x32 shared memory, 16x16 threads blocks)
  // each threads loads four values from global memory into shared mem
  // if in image area, get value in global mem, else 0
  int x, y; // image based coordinate

  // original image based coordinate
  const int x0 = threadIdx.x + (blockIdx.x * blockDim.x);
  const int y0 = threadIdx.y + (blockIdx.y * blockDim.y);

  // case1: upper left
  x = x0 - n;
  y = y0 - n;
  if ( x < 0 || y < 0 )
    s_data[threadIdx.x][threadIdx.y] = 0;
  else
    s_data[threadIdx.x][threadIdx.y] = data[ pos - n - (width * n)];

  // case2: upper right
  x = x0 + n;
  y = y0 - n;
  if ( x > width - 1 || y < 0 )
    s_data[threadIdx.x + blockDim.x][threadIdx.y] = 0;
  else
    s_data[threadIdx.x + blockDim.x][threadIdx.y] = data[pos + n - (width * n)];

  // case3: lower left
  x = x0 - n;
  y = y0 + n;
  if (x < 0 || y > height - 1)
    s_data[threadIdx.x][threadIdx.y + blockDim.y] = 0;
  else
    s_data[threadIdx.x][threadIdx.y + blockDim.y] = data[pos - n + (width * n)];

  // case4: lower right
  x = x0 + n;
  y = y0 + n;
  if ( x > width - 1 || y > height - 1)
    s_data[threadIdx.x + blockDim.x][threadIdx.y + blockDim.y] = 0;
  else
    s_data[threadIdx.x + blockDim.x][threadIdx.y + blockDim.y] = data[pos + n + (width * n)];

  __syncthreads();

  // convolution
  int sum = 0;
  x = n + threadIdx.x;
  y = n + threadIdx.y;
  for (int i = - n; i <= n; i++)
    for (int j = - n; j <= n; j++)
      sum += s_data[x + i][y + j] * SOBEL[n + j] * SOBEL[n + i];

  result[pos] = clamp(sum);
}

void convolution_con_tiled (unsigned char *data, unsigned char *result, int width, int height){
  unsigned char *d_data, *d_result;

  cudaMalloc(&d_data, sizeof(unsigned char) * width * height);
  cudaMalloc(&d_result, sizeof(unsigned char) * width * height);

  cudaMemcpy(d_data, data, sizeof(unsigned char) * width * height, cudaMemcpyHostToDevice);

  dim3 dimGrid (ceil (width / float (BLOCK_SIZE)), ceil (height / float (BLOCK_SIZE)), 1);
  dim3 dimBlock (BLOCK_SIZE, BLOCK_SIZE, 1);

  convolution_kernel_tiled<<<dimGrid, dimBlock>>> (d_data, d_result, width, height);

  cudaMemcpy(result, d_result, sizeof(unsigned char) * width * height, cudaMemcpyDeviceToHost);

  cudaFree(d_data);
  cudaFree(d_result);
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
  cudaMemcpy(result, d_result, sizeof(unsigned char) * width * height, cudaMemcpyDeviceToHost);

  cudaFree(d_data);
  cudaFree(d_result);
}

int main (int argc, char **argv){
  if (argc < 2){
    cout << "Usage: ./concolution2d image_name" << endl;
  }

  char mask[] = {-1,0,1,-2,0,2,-1,0,1};
  cudaMemcpyToSymbol (SOBEL, mask, 3 * 3 * sizeof (char));

  Mat image, result;
  image = imread("../img1.jpg",1);
  //imwrite ("miau.jpg", image);

  Size s = image.size();
  int Row = s.width;
  int Col = s.height;

  unsigned char * In = (unsigned char*)malloc (sizeof (unsigned char)* Row * Col * image.channels());
  unsigned char *Gray = (unsigned char*)malloc (sizeof (unsigned char)* Row * Col);
  unsigned char * Out = (unsigned char*)malloc (sizeof (unsigned char)* Row * Col);

  In = image.data;

  img2gray (In, Gray, Row, Col);
  result.create(Row, Col, CV_8UC1);
  result.data = Out;
  imwrite ("miau.jpg", result);



  //convolution_con_tiled (Gray, Out, Row, Col);



  return 0;
}
