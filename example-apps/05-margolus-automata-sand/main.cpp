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
#include <stb_image_write.h>

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
  
  //Grid grid = gridCreate(400, 200); // 60 FPS. (Probably would have been faster if not G-Sync)
  GridGPU gridGpu = gridGpuCreate(8'192, 4'096);

  //Grid grid = gridCreate(2400, 1200); // 30 FPS
  launchGridClear(gridGpu);
  cudaOnErrorPrintAndExit();

  //gridResetBoundaries(grid);
  launchGridResetBoundaries(gridGpu);
  cudaOnErrorPrintAndExit();

  curandState* d_randState;
  cudaMalloc(&d_randState, sizeof(curandState) * gridGpu.size2.x * gridGpu.size2.y);
  launchSeedRandomization(d_randState, gridGpu);
  cudaOnErrorPrintAndExit();

  const int nRnd = 40000;
  //const int nRnd = 1'440'000;
  //gridAddRandomCells(grid, nRnd);
  launchGridAddRandomCells(d_randState, gridGpu, 0.05f);
  cudaOnErrorPrintAndExit();
  
  glm::uvec2 winSize = workshop.getWindowSize();
  double aspectRatio = static_cast<double>(winSize.x) / winSize.y;

  uint32_t texInitHeight = 80;
  uint32_t texInitWidth = texInitHeight * aspectRatio;
  std::vector<uint32_t> pixels(texInitWidth * texInitHeight);

  auto desc = ws::Texture::Specs{texInitWidth, texInitHeight, ws::Texture::Format::RGBA8, ws::Texture::Filter::Nearest};
  ws::Texture tex{desc};

  struct cudaGraphicsResource* texCuda{};
  cudaGraphicsGLRegisterImage(&texCuda, tex.getId(), GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore);

  // VAO binding is needed in 4.6 was not needed in 3.1
  uint32_t vao;
  glGenVertexArrays(1, &vao);

  while (!workshop.shouldStop()) {
    workshop.beginFrame();
    
    workshop.drawUI();

    ImGui::Begin("Main");
    ImGui::Separator();
    if (ImGui::Button("Regenerate")) {
      //gridClear(grid);
      //gridAddRandomCells(grid, nRnd);
      launchGridClear(gridGpu);
      launchGridAddRandomCells(d_randState, gridGpu, 0.05f);
      launchGridResetBoundaries(gridGpu);
    }
    if (ImGui::Button("RulesDebugger - Save 2 States"))
      rulesDebugger.saveSuccessiveStatesToImages(2);

    static int32_t offsetX = 20;
    static int32_t offsetY = 60;
    ImGui::DragInt("offsetX", &offsetX, 1, 0, gridGpu.size2.x - tex.specs.width);
    ImGui::DragInt("offsetY", &offsetY, 1, 0, gridGpu.size2.y - tex.specs.height);
    assert(tex.specs.width + offsetX <= gridGpu.size2.x);
    assert(tex.specs.height + offsetY <= gridGpu.size2.y);
    int32_t w = tex.specs.width;
    int32_t h = tex.specs.height;
    if (ImGui::DragInt("texWidth", &w, 1, 0, gridGpu.size2.x - offsetX) || ImGui::DragInt("texHeight", &h, 1, 0, gridGpu.size2.y - offsetY)) {
      if (texCuda)
        cudaGraphicsUnregisterResource(texCuda);
      tex.resize(w, h);
      cudaGraphicsGLRegisterImage(&texCuda, tex.getId(), GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore);
    }

    ImDrawList* drawList = ImGui::GetWindowDrawList();
    const ImVec2 cursor = ImGui::GetCursorScreenPos();
    const float gridRectHeight = 200;
    const ImVec2 gridRectSize {(float)gridGpu.size2.x / gridGpu.size2.y * gridRectHeight, gridRectHeight};
    const float texRectHeight = gridRectHeight / gridGpu.size2.y * tex.specs.height;
    const ImVec2 texRectSize{(float)tex.specs.width / tex.specs.height * texRectHeight, texRectHeight};
    const ImVec2 texRectOffset{(float)offsetX * gridRectSize.x / gridGpu.size2.x, (float)offsetY * gridRectSize.y / gridGpu.size2.y};
    drawList->AddRect(cursor, {cursor.x + gridRectSize.x, cursor.y + gridRectSize.y}, 0xFF00FFFF);
    drawList->AddRect({cursor.x + texRectOffset.x, cursor.y + texRectOffset.y}, {cursor.x + texRectOffset.x + texRectSize.x, cursor.y + texRectOffset.y + texRectSize.y}, 0xFFFFFFFF);
    ImGui::Dummy(gridRectSize);
    ImGui::Separator();
    static bool shouldShowImGuiDemo = false;
    ImGui::Checkbox("Show Demo", &shouldShowImGuiDemo);
    if (shouldShowImGuiDemo)
      ImGui::ShowDemoWindow();
    ImGui::End();

    if (workshop.getFrameNo() % 1 == 0) {
      //gridApplyRulesToGrid(grid);
      launchGridApplyRulesToGrid(gridGpu);
      cudaDeviceSynchronize();
      cudaOnErrorPrintAndExit();

      //gridGpuDownload(gridGpu, grid);
    }

    // Copy part of it into texture
    /*
    for (uint32_t i = 0; i < extend.y; ++i) {
      for (uint32_t j = 0; j < extend.x; ++j) {
        //uint32_t gix = (i + offset.y) * grid.size2.x + (j + offset.x);
        uint32_t gix = ((extend.y - 1 - i) + offset.y) * grid.size2.x + (j + offset.x); // upside-down
        uint32_t pix = i * extend.x + j;
        uint32_t color = 0xFF000000;
        if (grid.data[gix] == 1)
          color = 0xFF00FFFF;
        else if (grid.data[gix] == 2)
          color = 0xFF0000FF;
        else if (grid.data[gix] == 3)
          color = 0xFFFF0000;
        pixels[pix] = color;
      }
    }
    tex.loadPixels(pixels.data());
    */

    cudaGraphicsMapResources(1, &texCuda, 0);
    cudaArray_t texArray;
    cudaGraphicsSubResourceGetMappedArray(&texArray, texCuda, 0, 0);

    struct cudaResourceDesc resDesc{};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = texArray;
    cudaSurfaceObject_t surface = 0;
    cudaCreateSurfaceObject(&surface, &resDesc);

    // TODO: grid cells should be allocated via cudaMalloc3D.
    // This might require updating the kernel logics
    // Then copy subsection of cells into texArray
    // Are there functions to query a cudaArray_t ? such as its dimensions...
    launchGridCopyToTexture(gridGpu, surface, offsetX, offsetY, tex.specs.width, tex.specs.height);
    cudaDeviceSynchronize();
    cudaOnErrorPrintAndExit();

    cudaDestroySurfaceObject(surface);
    cudaGraphicsUnmapResources(1, &texCuda, 0);

    glClearColor(1, 0, 1, 1);
    glClear(GL_COLOR_BUFFER_BIT);

    const auto winSize = workshop.getWindowSize();
    glViewport(0, 0, winSize.x, winSize.y);

    shader.bind();
    tex.bind();
    glBindVertexArray(vao);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    glBindVertexArray(0);
    ws::Texture::unbind();
    shader.unbind();

    workshop.endFrame();
  }

  cudaGraphicsUnregisterResource(texCuda);

  return 0;
}