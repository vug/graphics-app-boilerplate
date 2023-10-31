#include "kernels.h"

#include <cuComplex.h>

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

__global__ void genMandelbrot(cudaSurfaceObject_t surf, int texWidth, int texHeight, Model model, int maxIter, bool useDouble, int timeStep) {
  int texX = blockIdx.x * blockDim.x + threadIdx.x;
  int texY = blockIdx.y * blockDim.y + threadIdx.y;
  if (texX >= texWidth || texY >= texHeight)
    return;

  double u = (double)texX / (double)texWidth - 0.5; // [0, 1)
  double v = (double)texY / (double)texHeight - 0.5; // [0, 1)
  double width = model.height / texHeight * texWidth;
  double x = model.topLeft.x + u * width;
  double y = model.topLeft.y + v * model.height;
  bool bounded = true;
  int nSteps = 0;
  if (useDouble) {
    cuDoubleComplex z0 = make_cuDoubleComplex(model.z0.x, model.z0.y);
    cuDoubleComplex coord = make_cuDoubleComplex(x, y);
    cuDoubleComplex z = model.fractalType == Fractal_Mandelbrot ? z0 : coord;
    for (int i = 0; i < maxIter; i++) {
      z = model.fractalType == Fractal_Mandelbrot ? cuCadd(cuCmul(z, z), coord) : cuCadd(cuCmul(z, z), z0);
      ++nSteps;
      if (cuCabs(z) > 2.) {
        bounded = false;
        break;
      }
    }
  } else {
    cuFloatComplex z0 = make_cuFloatComplex(model.z0.x, model.z0.y);
    cuFloatComplex coord = make_cuFloatComplex(x, y);
    cuFloatComplex z = model.fractalType == Fractal_Mandelbrot ? z0 : coord;
    for (int i = 0; i < maxIter; i++) {
      z = model.fractalType == Fractal_Mandelbrot ? cuCaddf(cuCmulf(z, z), coord) : cuCaddf(cuCmulf(z, z), z0);
      ++nSteps;
      if (cuCabsf(z) > 2.) {
        bounded = false;
        break;
      }
    }
  }

  unsigned char val = bounded ? 0 : 255 * (maxIter - nSteps) / maxIter;
  unsigned char red = val;
  unsigned char green = val;
  unsigned char blue = val;
  unsigned char alpha = val;
  uchar4 pixel{red, green, blue, alpha};
  surf2Dwrite(pixel, surf, texX * sizeof(uchar4), texY);
}

void launchGenMandelbrot(cudaSurfaceObject_t surf, int texWidth, int texHeight, Model model, int maxIter, bool useDouble, int timeStep) {
  const auto threadSize = dim3(32, 32);
  const auto blockSize = dim3(texWidth / threadSize.x + 1, texHeight / threadSize.y + 1);
  genMandelbrot<<<blockSize, threadSize>>>(surf, texWidth, texHeight, model, maxIter, useDouble, timeStep);
}