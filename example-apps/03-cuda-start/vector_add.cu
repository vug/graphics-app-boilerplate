#include "vector_add.h"

__global__ void vectorAdd(float* a, float* b, float* out) {
  size_t ix = threadIdx.x;
  out[ix] = a[ix] + b[ix];
}