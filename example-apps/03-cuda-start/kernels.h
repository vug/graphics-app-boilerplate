#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <glm/vec2.hpp>

enum FractalType : int {
  Fractal_Mandelbrot = 0,
  Fractal_Julia = 1,
};

struct Model {
  glm::vec2 topLeft{-0.4f, 0.f};
  float height = 2.25f;
  glm::vec2 z0{0.f, 0.f};
  int fractalType = Fractal_Mandelbrot; // Mandelbrot or Julia
};


__global__ void vectorAdd(float* a, float* b, float* out);

__global__ void genTexture(unsigned int* pixels, int width, int height);
void launchGenTexture(unsigned int* pixels, int width, int height);

__global__ void genSurface(cudaSurfaceObject_t surf, int width, int height, int timeStep);
void launchGenSurface(cudaSurfaceObject_t surf, int width, int height, int timeStep);

__global__ void genMandelbrot(cudaSurfaceObject_t surf, int texWidth, int texHeight, Model model, int maxIter, bool useDouble, int timeStep);
void launchGenMandelbrot(cudaSurfaceObject_t surf, int texWidth, int texHeight, Model model, int maxIter, bool useDouble, int timeStep);