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

void launchGenTexture(unsigned int* pixels, int width, int height) {
  const auto threadSize = dim3(32, 32);
  const auto blockSize = dim3(width / threadSize.x + 1, height / threadSize.y + 1);
  printf("numPixels: %d, sizeBytes: %zd, blockSize (%d, %d), threadSize: (%d, %d)\n", width * height, width * height * sizeof(unsigned int), blockSize.x, blockSize.y, threadSize.x, threadSize.y);
  genTexture<<<blockSize, threadSize>>>(pixels, width, height);
}

__global__ void genSurface(cudaSurfaceObject_t surf, int width, int height, int timeStep) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height)
    return;

  unsigned char red = ((x + timeStep) % width) * 255 / width;
  unsigned char green = y * 255 / height;
  unsigned char blue = 0;
  unsigned char alpha = 255;
  uchar4 pixel{red, green, blue, alpha};
  surf2Dwrite(pixel, surf, x * sizeof(uchar4), y);  // TODO: learn why x * 4 but just y
}

void launchGenSurface(cudaSurfaceObject_t surf, int width, int height, int timeStep) {
  const auto threadSize = dim3(32, 32);
  const auto blockSize = dim3(width / threadSize.x + 1, height / threadSize.y + 1);
  genSurface<<<blockSize, threadSize>>>(surf, width, height, timeStep);
}