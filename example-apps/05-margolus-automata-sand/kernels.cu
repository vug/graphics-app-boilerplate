#include "kernels.h"

#include <cuComplex.h>

#include <stdio.h>

__global__ void gridClear(GridGPU g) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x == 0 && y == 0)
    printf("gridClear. Hi from first thread!\n");

  if (x >= g.size2.x || y >= g.size2.y)
    return;

  const int ix = y * g.size2.x + x;
  g.cells[ix] = 0x000000;
}

void launchGridClear(GridGPU& g) {
  const auto threadSize = dim3(32, 32);
  const auto blockSize = dim3(g.size2.x / threadSize.x + 1, g.size2.y / threadSize.y + 1);
  printf("Launching gridClear: %p blockSize (%d, %d), threadSize (%d, %d)\n", g.cells, blockSize.x, blockSize.y, threadSize.x, threadSize.y);
  gridClear<<<blockSize, threadSize>>>(g);
}

__global__ void gridResetBoundaries(GridGPU g) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x == 0 && y == 0)
    printf("gridResetBoundaries. Hi from first thread!\n");


  if (x >= g.size2.x || y >= g.size2.y)
    return;

  const int ix = y * g.size2.x + x;

  // Side walls: By trial and error discovered that this "dashed" pattern acts like a wall
  // ` = 1` results in creation of particles because boundaries are reset after iterations
  if (x == 0 || x == g.size2.x - 1)
    g.cells[ix] = y % 2;
  if (y == 0)
    g.cells[ix] = 0;
  if (y == g.size2.y - 1)
    g.cells[ix] = 1;
}

void launchGridResetBoundaries(GridGPU& g) {
  const auto threadSize = dim3(32, 32);
  const auto blockSize = dim3(g.size2.x / threadSize.x + 1, g.size2.y / threadSize.y + 1);
  printf("Launching GridResetBoundaries: %p blockSize (%d, %d), threadSize (%d, %d)\n", g.cells, blockSize.x, blockSize.y, threadSize.x, threadSize.y);
  gridResetBoundaries<<<blockSize, threadSize>>>(g);
}

__global__ void setupKernel(curandState_t* state) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  curand_init(1234, idx, 0, &state[idx]);
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