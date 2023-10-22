#include "kernels.h"
#include <stdio.h>

__global__ void vectorAdd(float* a, float* b, float* out) {
  size_t ix = threadIdx.x;
  out[ix] = a[ix] + b[ix];
}

__global__ void genTexture(unsigned int* pixels, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int idx = y * width + x;

  if (x >= width || y >= height)
    return;

  unsigned char red = x * 255 / width;
  unsigned char green = y * 255 / height;
  unsigned char blue = 0;
  unsigned char alpha = 255;
  pixels[idx] = (alpha << 24) + (blue << 16) + (green << 8) + red;

  if (idx >= 480000) {
    printf("r: (%d, %d), idx: %d, block: (%d, %d), thread: (%d %d), %X\n", x, y, idx, blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, pixels[idx]);
  }
}