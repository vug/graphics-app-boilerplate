#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void vectorAdd(float* a, float* b, float* out);