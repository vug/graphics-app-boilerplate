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
