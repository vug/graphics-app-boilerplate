// Main file had to be main.cu and not main.cpp otherwise <<< is not recognized.
// I was able to overcome this in VS Code by compiling CU files via NVCC into a library and compile main.cpp via MSVC and link them later
#include "kernels.h"

#include <Workshop/Shader.hpp>
#include <Workshop/Texture.hpp>
#include <Workshop/Workshop.hpp>

#include <cuda_gl_interop.h>
#include <glad/gl.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <glm/gtc/type_ptr.hpp>
#include <glm/vec3.hpp>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

// #include <implot.h>
// #include <tiny_obj_loader.h>
// #include <vivid/vivid.h>

#include <iostream>
#include <print>
#include <random>
#include <ranges>

void cudaOnErrorPrintAndExit() {
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
}

struct Grid {
  std::vector<uint32_t> data{};
  glm::uvec2 size{};
  glm::uvec2 size2{};
};

Grid gridCreate(uint32_t nx, uint32_t ny) {
  Grid g;
  g.size = {nx, ny}; // simulation area
  g.size2 = {nx + 2, ny + 2}; // simulation area + boundaries
  const uint32_t numCells = g.size2.y * g.size2.x;
  g.data.resize(numCells, 0);
  return g;
}

GridGPU gridGpuCreate(uint32_t width, uint32_t height) {
  GridGPU g{
    .cells = {},
    .size = {width, height},
    .size2 = {width + 2, height + 2},
  };
  // TODO: use cudaMalloc3D
  cudaMalloc(&g.cells, sizeof(uint32_t) * g.size2.x * g.size2.y);
  return g;
}

void gridGpuUpload(const Grid& gCpuSrc, GridGPU& gGpuDst) {
  assert(gGpuDst.cells != nullptr);
  assert(gGpuDst.size.x == gCpuSrc.size.x && gGpuDst.size.y == gCpuSrc.size.y);
  cudaMemcpy(gGpuDst.cells, gCpuSrc.data.data(), sizeof(uint32_t) * gCpuSrc.size2.x * gCpuSrc.size2.y, cudaMemcpyDefault);
}

void gridGpuDownload(const GridGPU& gGpuSrc, Grid& gCpuDst) {
  assert(gGpuSrc.cells != nullptr);
  assert(gGpuSrc.size.x == gCpuDst.size.x && gGpuSrc.size.y == gCpuDst.size.y);
  cudaMemcpy(gCpuDst.data.data(), gGpuSrc.cells, sizeof(uint32_t) * gCpuDst.size2.x * gCpuDst.size2.y, cudaMemcpyDefault);
}

void gridResetBoundaries(Grid& grid) {
  // Side walls: By trial and error discovered that this "dashed" pattern acts like a wall
  // ` = 1` results in creation of particles because boundaries are reset after iterations
  for (int i = 0;  i < grid.size2.y; ++i) {
    const size_t ixLeft = i * grid.size2.x;
    const size_t ixRight = ixLeft + grid.size2.x - 1;
    grid.data[ixLeft] = grid.data[ixRight]  = i % 2;
  }
  for (int j = 0; j < grid.size2.x; j++) {
    const size_t ixTop = j;
    const size_t ixBottom = (grid.size2.y - 1) * grid.size2.x + j;
    grid.data[ixTop] = 0; // (0, j)
    grid.data[ixBottom] = 1; // (ny + 1, j)
  }
}

void gridClear(Grid& grid) {
  const size_t numCells = grid.size2.y * grid.size2.x;
  for (size_t ix = 0; ix < numCells; ++ix)
    grid.data[ix] = 0;
  gridResetBoundaries(grid);
}

void gridAddRandomCells(Grid& grid, uint32_t numRandomCells) {
  std::random_device rndDev;
  std::mt19937 gen(rndDev());
  std::uniform_int_distribution<> uDistX(1, grid.size2.x - 2);
  std::uniform_int_distribution<> uDistY(1, grid.size2.y - 2);
  for (int i = 0; i < numRandomCells; ++i)
    grid.data[uDistY(gen) * grid.size2.x + uDistX(gen)] = 1;  // (rndX, rndY)
}

uint32_t gridGetBlockCase(const Grid& grid, uint32_t i, uint32_t j) {
  // clock-wise from north-west / top-left of 2x2 block
  const size_t offsetTopLeft = i * grid.size2.x + j; 
  const size_t offsetTopRight = i * grid.size2.x + (j + 1); 
  const size_t offsetBottomRight = (i + 1) * grid.size2.x + (j + 1); 
  const size_t offsetBottomLeft = (i + 1) * grid.size2.x + j;
  const uint32_t topLeft = grid.data[offsetTopLeft];
  const uint32_t topRight = grid.data[offsetTopRight];
  const uint32_t bottomRight = grid.data[offsetBottomRight];
  const uint32_t bottomLeft = grid.data[offsetBottomLeft];
  uint32_t result = (topLeft << 3) + (topRight << 2) + (bottomRight << 1) + (bottomLeft << 0);
  return result;
}

void gridSetBlockCase(Grid& grid, uint32_t i, uint32_t j, uint32_t blockCase) {
  const size_t offsetTopLeft = i * grid.size2.x + j;
  const size_t offsetTopRight = i * grid.size2.x + (j + 1);
  const size_t offsetBottomRight = (i + 1) * grid.size2.x + (j + 1);
  const size_t offsetBottomLeft = (i + 1) * grid.size2.x + j;
  const uint32_t topLeft = (blockCase & 0b1000) == 0b1000;
  const uint32_t topRight = (blockCase & 0b0100) == 0b0100;
  const uint32_t bottomRight = (blockCase & 0b0010) == 0b0010;
  const uint32_t bottomLeft = (blockCase & 0b0001) == 0b0001;
  //const uint32_t topLeft = (blockCase & 0b0001) == 0b0001;
  //const uint32_t topRight = (blockCase & 0b0010) == 0b0010;
  //const uint32_t bottomRight = (blockCase & 0b0100) == 0b0100;
  //const uint32_t bottomLeft = (blockCase & 0b1000) == 0b1000;
  grid.data[offsetTopLeft] = topLeft;
  grid.data[offsetTopRight] = topRight;
  grid.data[offsetBottomRight] = bottomRight;
  grid.data[offsetBottomLeft] = bottomLeft;
}

uint32_t gridApplyRulesToCase(uint32_t oldCase) {
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
    default:
      assert(false); // no such case should happen
      throw;
  }
}

void gridApplyRulesToBlock(Grid& grid, uint32_t i, uint32_t j) {
  uint32_t oldCase = gridGetBlockCase(grid, i, j);
  uint32_t newCase = gridApplyRulesToCase(oldCase);
  gridSetBlockCase(grid, i, j, newCase);
}

void gridApplyRulesToGrid(Grid& grid) {
  for (int32_t i = grid.size.y - 1; i >= 1; i -= 2) {
    for (int32_t j = 1; j < grid.size.x; j += 2) {
      gridApplyRulesToBlock(grid, i, j);
    }
  }
  gridResetBoundaries(grid);

  for (int32_t i = grid.size.y; i >= 0; i -= 2) {
    for (int32_t j = 0; j < grid.size.x + 1; j += 2) {
      gridApplyRulesToBlock(grid, i, j);
    }
  }
  gridResetBoundaries(grid);
}

class RulesDebugger {
 public:
  Grid g;

  RulesDebugger() {
    g.size = {32, 2};
    g.size2 = {32, 2};
    resetData();
  }

  void applyRules() {
    for (int j = 0; j < 32; j += 2)
      gridApplyRulesToBlock(g, 0, j);
    ++stateNo;
  }

  void resetData() {
    g.data = {
      0, 0,   0, 0,   0, 0,   0, 0,   0, 1,   0, 1,   0, 1,   0, 1,   1, 0,   1, 0,   1, 0,   1, 0,   1, 1,   1, 1,   1, 1,   1, 1,
      0, 0,   0, 1,   1, 0,   1, 1,   0, 0,   0, 1,   1, 0,   1, 1,   0, 0,   0, 1,   1, 0,   1, 1,   0, 0,   0, 1,   1, 0,   1, 1,
    };
  }

  void saveStateToImage() {
    const std::string filename = std::format("state-{:02d}.png", stateNo);
    const auto img = g.data 
      | std::views::transform([](int32_t x) -> uint8_t { return x * 255; }) 
      | std::ranges::to<std::vector>();
    stbi_write_png(filename.c_str(), g.size.x, g.size.y, 1, img.data(), sizeof(uint8_t) * g.size.x);
  }

  void saveSuccessiveStatesToImages(int numStates) {
    resetData();
    for (int i = 0; i < numStates; ++i) {
      saveStateToImage();
      applyRules();
    }
  }

private:
  uint32_t stateNo = 0;

};


int main(int argc, char* argv[]) {
  std::cout << "Hi!\n";
  ws::Workshop workshop{1200, 600, "Sand 'Simulation' via Margolus Neighborhood Automata"};

  const char *vertexShader = R"(
#version 460

out VertexData {
  vec2 uv;
} v; // vertex-to-fragment or vertex-to-geometry

vec2 positions[4] = vec2[](vec2(-1, -1), vec2(1, -1), vec2(1, 1), vec2(-1, 1));
vec2 uvs[4] = vec2[](vec2(0, 0), vec2(1, 0), vec2(1, 1), vec2(0, 1));
int indices[6] = int[](0, 1, 2, 0, 2, 3);

void main () {
  int ix = indices[gl_VertexID];
	gl_Position = vec4 (positions[ix], 0.0, 1.0);
	v.uv = uvs[ix];
}
  )";

  const char *fragmentShader = R"(
#version 460

in VertexData {
  vec2 uv;
} v;

uniform sampler2D screenTexture;

layout (location = 0) out vec4 outColor;

void main () {
  //outColor = vec4(v.uv.x, v.uv.y, 0, 1.0); 

  vec3 tex =  texture(screenTexture, v.uv).rgb;
  outColor.rgb = tex;

  //float val = (tex.r + tex.g + tex.b) / 3.0;
  //outColor.rgb = vec3(val);
}
  )";
  ws::Shader shader{vertexShader, fragmentShader};
  RulesDebugger rulesDebugger{};
  
  Grid grid = gridCreate(400, 200); // 60 FPS. (Probably would have been faster if not G-Sync)
  GridGPU gridGpu = gridGpuCreate(400, 200);

  const int nRnd = 40000;
  //Grid grid = gridCreate(2400, 1200); // 30 FPS
  //const int nRnd = 1'440'000;

  //gridResetBoundaries(grid);
  launchGridClear(gridGpu);
  cudaDeviceSynchronize();
  cudaOnErrorPrintAndExit();
  launchGridResetBoundaries(gridGpu);
  cudaDeviceSynchronize();
  cudaOnErrorPrintAndExit();
  gridGpuDownload(gridGpu, grid);

  // TODO: continue to move functionality to the GPU
  gridAddRandomCells(grid, nRnd);

  glm::uvec2 winSize = workshop.getWindowSize();
  double aspectRatio = static_cast<double>(winSize.x) / winSize.y;

  uint32_t texHeight = grid.size2.y;
  uint32_t texWidth = texHeight * aspectRatio;
  glm::uvec2 offset{0, 0};
  glm::uvec2 extend{std::min(texWidth, grid.size2.x - offset.x), std::min(texHeight, grid.size2.y - offset.x)};
  std::vector<uint32_t> pixels(extend.x * extend.y);

  auto desc = ws::Texture::Specs{extend.x, extend.y, ws::Texture::Format::RGBA8, ws::Texture::Filter::Nearest};
  ws::Texture tex{desc};

  //struct cudaGraphicsResource* texCuda{};
  //cudaGraphicsGLRegisterImage(&texCuda, tex.getId(), GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore);
  // VAO binding is needed in 4.6 was not needed in 3.1
  uint32_t vao;
  glGenVertexArrays(1, &vao);

  while (!workshop.shouldStop()) {
    workshop.beginFrame();
    
    //winSize = workshop.getWindowSize();
    //if (tex.specs.width != winSize.x || tex.specs.height != winSize.y) {
    //  //if (texCuda) cudaGraphicsUnregisterResource(texCuda);
    //  //tex.resize(winSize.x, winSize.y);
    //  //cudaGraphicsGLRegisterImage(&texCuda, tex.getId(), GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore);
    //}

    ImGui::Begin("Main");
    ImGui::Text("Frame No: %6d, Frame Dur: %.2f, FPS: %.1f", workshop.getFrameNo(), workshop.getFrameDurationMs(), workshop.getFrameRate());
    bool vSync = workshop.getVSync();
    if (ImGui::Checkbox("VSync", &vSync))
      workshop.setVSync(vSync);
    ImGui::Separator();
    static glm::vec3 bgColor{42 / 256.0, 96 / 256.0, 87 / 256.0};
    ImGui::ColorEdit3("BG Color", glm::value_ptr(bgColor));
    ImGui::Separator();
    if (ImGui::Button("Regenerate")) {
      gridClear(grid);
      gridAddRandomCells(grid, nRnd);
    }
    if (ImGui::Button("RulesDebugger - Save 2 States"))
      rulesDebugger.saveSuccessiveStatesToImages(2);
    ImGui::Separator();
    static bool shouldShowImGuiDemo = false;
    ImGui::Checkbox("Show Demo", &shouldShowImGuiDemo);
    if (shouldShowImGuiDemo)
      ImGui::ShowDemoWindow();
    ImGui::End();

    if (workshop.getFrameNo() % 1 == 0)
      gridApplyRulesToGrid(grid);

  // Copy part of it into texture
    for (uint32_t i = 0; i < extend.y; ++i) {
      for (uint32_t j = 0; j < extend.x; ++j) {
        //uint32_t gix = (i + offset.y) * grid.size2.x + (j + offset.x);
        uint32_t gix = ((extend.y - 1 - i) + offset.y) * grid.size2.x + (j + offset.x); // upside-down
        uint32_t pix = i * extend.x + j;
        pixels[pix] = grid.data[gix] * 0xFF00FFFF;
      }
    }
    tex.loadPixels(pixels.data());

    //cudaGraphicsMapResources(1, &texCuda, 0);
    //cudaArray_t array;
    //cudaGraphicsSubResourceGetMappedArray(&array, texCuda, 0, 0);
    //struct cudaResourceDesc resDesc{};
    //resDesc.resType = cudaResourceTypeArray;
    //resDesc.res.array.array = array;
    //cudaSurfaceObject_t surface = 0;
    //cudaCreateSurfaceObject(&surface, &resDesc);

    //launchGenSurface(surface, winSize.x, winSize.y, timeStep++);

    //cudaDestroySurfaceObject(surface);
    //cudaGraphicsUnmapResources(1, &texCuda, 0);

    glClearColor(bgColor.x, bgColor.y, bgColor.z, 1);
    glClear(GL_COLOR_BUFFER_BIT);

    const auto winSize = workshop.getWindowSize();
    glViewport(0, 0, winSize.x, winSize.y);

    shader.bind();
    tex.bind();
    glBindVertexArray(vao);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    glBindVertexArray(0);
    tex.unbind();
    shader.unbind();

    workshop.endFrame();
  }

  //cudaGraphicsUnregisterResource(texCuda);

  return 0;
}