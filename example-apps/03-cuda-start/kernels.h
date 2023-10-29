#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void vectorAdd(float* a, float* b, float* out);

__global__ void genTexture(unsigned int* pixels, int width, int height);
void launchGenTexture(unsigned int* pixels, int width, int height);

__global__ void genSurface(cudaSurfaceObject_t surf, int width, int height, int timeStep);
void launchGenSurface(cudaSurfaceObject_t surf, int width, int height, int timeStep);

__global__ void genMandelbrot(cudaSurfaceObject_t surf, int texWidth, int texHeight, float x, float y, float height, int maxIter, int timeStep);
void launchGenMandelbrot(cudaSurfaceObject_t surf, int texWidth, int texHeight, float x, float y, float height, int maxIter, int timeStep);