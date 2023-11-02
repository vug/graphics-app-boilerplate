#include "kernels.h"

#include <cuComplex.h>

#include <stdio.h>

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