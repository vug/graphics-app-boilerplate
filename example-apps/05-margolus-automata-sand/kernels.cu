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

__global__ void seedRandomization(curandState_t* state, GridGPU g) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int ix = y * g.size2.x + x;
  if (x >= g.size2.x || y >= g.size2.y)
    return;

  const int globalSeed = 1234;
  const int localSequence = ix;
  curand_init(globalSeed, localSequence, 0, &state[ix]);
}

void launchSeedRandomization(curandState_t* state, const GridGPU& g) {
  const auto threadSize = dim3(32, 32);
  const auto blockSize = dim3(g.size2.x / threadSize.x + 1, g.size2.y / threadSize.y + 1);
  printf("Launching seedRandomization...");
  seedRandomization<<<blockSize, threadSize>>>(state, g);
}

__global__ void gridAddRandomCells(curandState_t* state, GridGPU g, float rate) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int ix = y * g.size2.x + x;
  if (x >= g.size2.x || y >= g.size2.y)
    return;

  float rndUniform = curand_uniform(state + ix);
  //const int upper = 10;
  //const int lower = 5;
  //rndUniform *= (upper - lower + 0.999999);
  //rndUniform += lower;
  //int rnd = (int)truncf(rndUniform);

  g.cells[ix] = rndUniform < rate ? 1 : 0;
}

void launchGridAddRandomCells(curandState_t* state, GridGPU g, float rate) {
  const auto threadSize = dim3(32, 32);
  const auto blockSize = dim3(g.size2.x / threadSize.x + 1, g.size2.y / threadSize.y + 1);
  printf("Launching gridAddRandomCells...");
  gridAddRandomCells<<<blockSize, threadSize>>>(state, g, rate);
}
