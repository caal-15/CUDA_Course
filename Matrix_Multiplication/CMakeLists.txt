cmake_minimum_required(VERSION 2.8)
find_package(CUDA QUIET REQUIRED)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -gencode arch=compute_30,code=sm_30)
cuda_add_executable(mul matrix_mul.cu)
