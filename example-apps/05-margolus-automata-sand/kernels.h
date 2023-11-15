#pragma once

#include "cuda_runtime.h"
#include <curand_kernel.h>
#include "device_launch_parameters.h"

struct GridGPU {
  unsigned int* cells;
  uint2 size;
  uint2 size2;
};

void launchGridClear(GridGPU& g);
void launchGridResetBoundaries(GridGPU& g);
__global__ void setupKernel(curandState_t* state);