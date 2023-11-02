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

// #include <implot.h>
// #include <stb_image.h>
// #include <tiny_obj_loader.h>
// #include <vivid/vivid.h>

#include <iostream>
#include <print>

void cudaOnErrorPrintAndExit() {
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
}

int main(int argc, char* argv[]) {
  std::cout << "Hi!\n";
  ws::Workshop workshop{800, 600, "Workshop App"};

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
  glm::uvec2 gridSize{10, 10};
  std::vector<uint32_t> grid(gridSize.x * gridSize.y);

  glm::uvec2 winSize = workshop.getWindowSize();
  double aspectRatio = static_cast<double>(winSize.x) / winSize.y;

  uint32_t texHeight = gridSize.x;
  uint32_t texWidth = texHeight * aspectRatio;
  glm::uvec2 offset{0, 0};
  glm::uvec2 extend{std::min(texWidth, gridSize.x - offset.x), std::min(texHeight, gridSize.y - offset.x)};
  std::vector<uint32_t> pixels(extend.x * extend.y);

  auto desc = ws::Texture::Specs{extend.x, extend.y, ws::Texture::Format::RGBA8, ws::Texture::Filter::Nearest};
  ws::Texture tex{desc};

  // Calculate the grid
  for (uint32_t i = 0; i < gridSize.y; ++i) {
    for (uint32_t j = 0; j < gridSize.x; ++j) {
      // Red: 0xFF0000FF, Green: 0xFF00FF00, Blue: 0xFFFF0000
      uint32_t color = 0xFFFFFFFF;
      if ((i/2 + j/2) % 2 == 0)
        color = 0xFFFFFF00; // teal
      else
        color = 0xFF00FFFF; // yellow
      grid[i * gridSize.x + j] = color;
    }
  }

  // Copy part of it into texture
  for (uint32_t i = 0; i < extend.y; ++i) {
    for (uint32_t j = 0; j < extend.x; ++j) {
      uint32_t gix = (i + offset.y) * gridSize.x + (j + offset.x);
      uint32_t pix = i * extend.x + j;
      pixels[pix] = grid[gix];
    }
  }
  tex.loadPixels(pixels.data());

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
    ImGui::Separator();
    static glm::vec3 bgColor{42 / 256.0, 96 / 256.0, 87 / 256.0};
    ImGui::ColorEdit3("BG Color", glm::value_ptr(bgColor));
    ImGui::Separator();
    static bool shouldShowImGuiDemo = false;
    ImGui::Checkbox("Show Demo", &shouldShowImGuiDemo);
    if (shouldShowImGuiDemo)
      ImGui::ShowDemoWindow();
    ImGui::End();

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