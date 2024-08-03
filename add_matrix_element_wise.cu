
#include <iostream>
#include <math.h>
// Kernel function to add the elements of two arrays
__global__
void add_matrix(int n, int m, float *x, float *y)
{
  int indexX = blockIdx.x * blockDim.x + threadIdx.x;
  int indexY = blockIdx.y * blockDim.y + threadIdx.y;
  int strideX = blockDim.x * gridDim.x;
  int strideY = blockDim.y * gridDim.y;
  for (int i = indexX; i < n; i += strideX) {
    for (int j = indexY; j < m; j += strideY) { 
      y[i * m + j] = x[i * m + j] + y[i * m + j];
    }
  }
}

int main(void)
{
  int N = 1<<10;
  int M = 1<<10;
  float *x, *y;

  cudaMallocManaged(&x, N * M * sizeof(float));
  cudaMallocManaged(&y, N * M * sizeof(float));

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < M; j++) {
      x[i * M + j] = 1.0f;
      y[i * M + j] = 2.0f;
    }
  }

  dim3 blockSize(16, 16);
  dim3 numBlocks((N + blockSize.x - 1) / blockSize.x, (M + blockSize.y - 1) / blockSize.y);

  add_matrix<<<numBlocks, blockSize>>>(N, M, x, y);

  cudaDeviceSynchronize();

  float maxError = 0.0f;
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < M; j++) {
      maxError = fmax(maxError, fabs(y[i * M + j] - 3.0f));
    }
  }

  std::cout << "Max error: " << maxError << std::endl;

  cudaFree(x);
  cudaFree(y);
  
  return 0;
}
