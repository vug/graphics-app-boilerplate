#include "kernels.h"

#include <cuComplex.h>

#include <stdio.h>

__global__ void gridClear(GridGPU g) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x == 0 && y == 0)
    //printf("gridClear. Hi from first thread!\n");

  if (x >= g.size2.x || y >= g.size2.y)
    return;

  const int ix = y * g.size2.x + x;
  g.cells[ix] = 0x000000;
}

void launchGridClear(GridGPU& g) {
  const auto threadSize = dim3(32, 32);
  const auto blockSize = dim3(g.size2.x / threadSize.x + 1, g.size2.y / threadSize.y + 1);
  //printf("Launching gridClear: %p blockSize (%d, %d), threadSize (%d, %d)\n", g.cells, blockSize.x, blockSize.y, threadSize.x, threadSize.y);
  gridClear<<<blockSize, threadSize>>>(g);
}

__global__ void gridResetBoundaries(GridGPU g) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  //if (x == 0 && y == 0)
  //  printf("gridResetBoundaries. Hi from first thread!\n");

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
  //printf("Launching GridResetBoundaries: %p blockSize (%d, %d), threadSize (%d, %d)\n", g.cells, blockSize.x, blockSize.y, threadSize.x, threadSize.y);
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
  //printf("Launching seedRandomization...\n");
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
  //printf("Launching gridAddRandomCells...\n");
  gridAddRandomCells<<<blockSize, threadSize>>>(state, g, rate);
}

__device__ unsigned int gridGetBlockCase(GridGPU& grid, unsigned int i, unsigned int j) {
  // clock-wise from north-west / top-left of 2x2 block
  const size_t offsetTopLeft = i * grid.size2.x + j;
  const size_t offsetTopRight = i * grid.size2.x + (j + 1);
  const size_t offsetBottomRight = (i + 1) * grid.size2.x + (j + 1);
  const size_t offsetBottomLeft = (i + 1) * grid.size2.x + j;
  const unsigned int topLeft = grid.cells[offsetTopLeft];
  const unsigned int topRight = grid.cells[offsetTopRight];
  const unsigned int bottomRight = grid.cells[offsetBottomRight];
  const unsigned int bottomLeft = grid.cells[offsetBottomLeft];
  unsigned int result = (topLeft << 3) + (topRight << 2) + (bottomRight << 1) + (bottomLeft << 0);
  return result;
}

__device__ void gridSetBlockCase(GridGPU& grid, unsigned int i, unsigned int j, unsigned int blockCase) {
  const unsigned int offsetTopLeft = i * grid.size2.x + j;
  const unsigned int offsetTopRight = i * grid.size2.x + (j + 1);
  const unsigned int offsetBottomRight = (i + 1) * grid.size2.x + (j + 1);
  const unsigned int offsetBottomLeft = (i + 1) * grid.size2.x + j;
  const unsigned int topLeft = (blockCase & 0b1000) == 0b1000;
  const unsigned int topRight = (blockCase & 0b0100) == 0b0100;
  const unsigned int bottomRight = (blockCase & 0b0010) == 0b0010;
  const unsigned int bottomLeft = (blockCase & 0b0001) == 0b0001;
  // const unsigned int topLeft = (blockCase & 0b0001) == 0b0001;
  // const unsigned int topRight = (blockCase & 0b0010) == 0b0010;
  // const unsigned int bottomRight = (blockCase & 0b0100) == 0b0100;
  // const unsigned int bottomLeft = (blockCase & 0b1000) == 0b1000;
  grid.cells[offsetTopLeft] = topLeft;
  grid.cells[offsetTopRight] = topRight;
  grid.cells[offsetBottomRight] = bottomRight;
  grid.cells[offsetBottomLeft] = bottomLeft;
}

__device__ unsigned int gridApplyRulesToCase2(unsigned int oldCase) {
  switch (oldCase) {
    case 0b0000:
      return 0b0000;
    case 0b0010:
      return 0b0010;
    case 0b0001:
      return 0b0001;
    case 0b0011:
      return 0b0011;
    case 0b0100:
      return 0b0010;
    case 0b0110:
      return 0b0011;
    case 0b0101:
      return 0b0011;
    case 0b0111:
      return 0b0111;
    case 0b1000:
      return 0b0001;
    case 0b1010:
      return 0b0011;
    case 0b1001:
      return 0b0011;
    case 0b1011:
      return 0b1011;
    case 0b1100:
      return 0b0011;
    case 0b1110:
      return 0b0111;
    case 0b1101:
      return 0b1011;
    case 0b1111:
      return 0b1111;
  }
}

__device__ void gridApplyRulesToBlock(GridGPU& grid, unsigned int i, unsigned int j) {
  unsigned int oldCase = gridGetBlockCase(grid, i, j);
  unsigned int newCase = gridApplyRulesToCase2(oldCase);
  gridSetBlockCase(grid, i, j, newCase);
}

__global__ void gridApplyRulesToGridA(GridGPU g) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= g.size2.x || y >= g.size2.y)
    return;

  if ((x % 2 == 0) && (y % 2 == 0)) {
    gridApplyRulesToBlock(g, y, x);
    //const int ix = y * g.size2.x + x;
    //g.cells[ix] = 2;
  }
}

__global__ void gridApplyRulesToGridB(GridGPU g) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= g.size.x || y >= g.size.y) // Note NOT size2. Only here!
    return;

  if ((x % 2 == 1) && (y % 2 == 1)) {
    gridApplyRulesToBlock(g, y, x);
    //const int ix = y * g.size2.x + x;
    //g.cells[ix] = 3;
  }
}

void launchGridApplyRulesToGrid(GridGPU g) {
  const auto threadSize = dim3(32, 32);
  const auto blockSize = dim3(g.size2.x / threadSize.x + 1, g.size2.y / threadSize.y + 1);
  gridApplyRulesToGridA<<<blockSize, threadSize>>>(g);
  gridResetBoundaries<<<blockSize, threadSize>>>(g);
  gridApplyRulesToGridB<<<blockSize, threadSize>>>(g);
  gridResetBoundaries<<<blockSize, threadSize>>>(g);
}

__global__ void gridCopyToTexture(GridGPU g, cudaSurfaceObject_t surface, unsigned int offsetX, unsigned int offsetY, unsigned int texWidth, unsigned int texHeight) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int ix = (offsetY + y) * g.size2.x + (offsetX + x);
  if (x >= texWidth || y >= texHeight)
    return;

  const int texX = x;
  const int texY = y;
  uchar4 pixel{0, 0, 0, 255}; // black
  if (g.cells[ix] == 1) {
    pixel = {255, 255, 0, 255};
  }
  surf2Dwrite(pixel, surface, texX * sizeof(uchar4), texHeight - 1 - texY);
}

void launchGridCopyToTexture(GridGPU g, cudaSurfaceObject_t surface, unsigned int offsetX, unsigned int offsetY, unsigned int texWidth, unsigned int texHeight) {
  const auto threadSize = dim3(32, 32);
  const auto blockSize = dim3(texWidth / threadSize.x + 1, texHeight / threadSize.y + 1);
  gridCopyToTexture<<<blockSize, threadSize>>>(g, surface, offsetX, offsetY, texWidth, texHeight);
}