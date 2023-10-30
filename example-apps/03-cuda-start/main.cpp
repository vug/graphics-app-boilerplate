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

void calcPixelsCpuToTex(ws::Texture& tex, const glm::uvec2& ws) {

  std::vector<uint32_t> pixels(ws.x * ws.y);
  for (uint32_t i = 0; i < ws.y; ++i) {
    for (uint32_t j = 0; j < ws.x; ++j) {
      // Red: 0xFF0000FF, Green: 0xFF00FF00, Blue: 0xFFFF0000
      const uint8_t red = j * 255 / ws.x;
      const uint8_t green = i * 255 / ws.y;
      const uint8_t blue = 0;
      const uint8_t alpha = 255;
      pixels[i * ws.x + j] = (alpha << 24) + (blue << 16) + (green << 8) + red;
    }
  }
  tex.loadPixels(pixels.data());
}

void calcPixelsGpuToCpuToTex(ws::Texture& tex, const glm::uvec2& ws) {
  std::vector<uint32_t> pixels(ws.x * ws.y);
  size_t texSizeBytes = pixels.size() * sizeof(uint32_t);
  uint32_t* d_pixels;
  cudaMalloc(&d_pixels, texSizeBytes);

  launchGenTexture(d_pixels, ws.x, ws.y);
  cudaDeviceSynchronize(); // necessary?
  cudaOnErrorPrintAndExit();

  cudaMemcpy(pixels.data(), d_pixels, texSizeBytes, cudaMemcpyDeviceToHost);
  cudaFree(d_pixels);

  tex.loadPixels(pixels.data());
}

void calcPixelsGlInterop(ws::Texture& tex, const glm::uvec2& ws) {
  struct cudaGraphicsResource* texCuda{};
  cudaGraphicsGLRegisterImage(&texCuda, tex.getId(), GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore);
  cudaOnErrorPrintAndExit();

  cudaGraphicsMapResources(1, &texCuda, 0);
  cudaOnErrorPrintAndExit();

  // Get the array from GL Texture Resource after each map (instead of creating and allocating one from scratch)
  cudaArray_t array;
  cudaGraphicsSubResourceGetMappedArray(&array, texCuda, 0, 0);
  cudaOnErrorPrintAndExit();

  //struct cudaResourceDesc resDesc{};
  struct cudaResourceDesc resDesc;
  //memset(&resDesc, 0, sizeof(resDesc)); // C API?
  resDesc.resType = cudaResourceTypeArray;
  resDesc.res.array.array = array;
  cudaSurfaceObject_t surface = 0;
  cudaCreateSurfaceObject(&surface, &resDesc);
  cudaOnErrorPrintAndExit();

  launchGenSurface(surface, ws.x, ws.y, 400);
  cudaOnErrorPrintAndExit();

  cudaDestroySurfaceObject(surface);
  cudaOnErrorPrintAndExit();

  cudaGraphicsUnmapResources(1, &texCuda, 0);
  cudaOnErrorPrintAndExit();

  cudaGraphicsUnregisterResource(texCuda);
  cudaOnErrorPrintAndExit();
}

struct Model {
  glm::vec2 topLeft{-0.4f, 0.f};
  float height = 2.25f;
};

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
  glm::uvec2 winSize = workshop.getWindowSize();
  auto desc = ws::Texture::Specs{winSize.x, winSize.y, ws::Texture::Format::RGBA8};
  ws::Texture tex{desc};

  //calcPixelsCpuToTex(tex, winSize);
  //calcPixelsGpuToCpuToTex(tex, winSize);
  //calcPixelsGlInterop(tex, winSize);

  struct cudaGraphicsResource* texCuda{};
  cudaGraphicsGLRegisterImage(&texCuda, tex.getId(), GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore);
  // VAO binding is needed in 4.6 was not needed in 3.1
  uint32_t vao;
  glGenVertexArrays(1, &vao);

  Model model{};
  glm::vec2 topLeftDragBegin{};

  workshop.onMouseMove = [](const glm::vec2 pos) { 
    //std::print("Cursor moved to ({}, {})\n", pos.x, pos.y); 
  };
  workshop.onMouseDragBegin = [&](const ws::MouseButton button, const glm::vec2& pos0, const glm::vec2& pos) {
    if (button != ws::MouseButton::MIDDLE)
      return;
    topLeftDragBegin = model.topLeft;
    //std::print("DRAG_BEGIN button {} from ({}, {}) to ({}, {})\n", static_cast<int>(button), pos0.x, pos0.y, pos.x, pos.y);
  };
  workshop.onMouseDragging = [&](const ws::MouseButton button, const glm::vec2& pos0, const glm::vec2& pos) {
    if (button != ws::MouseButton::MIDDLE)
      return;
    const glm::vec2 deltaPixels = pos - pos0;
    const glm::vec2& winSize = workshop.getWindowSize();
    const float distPerPixel = model.height / winSize.y;
    model.topLeft.x = topLeftDragBegin.x - distPerPixel * deltaPixels.x;
    model.topLeft.y = topLeftDragBegin.y + distPerPixel * deltaPixels.y;
    //std::print("DRAGGING button {} from ({}, {}) to ({}, {}), delta ({}, {})\n", static_cast<int>(button), pos0.x, pos0.y, pos.x, pos.y, deltaPixels.x, deltaPixels.y);
  };
  workshop.onMouseDragEnd = [](const ws::MouseButton button, const glm::vec2& pos0, const glm::vec2& pos) {
    if (button != ws::MouseButton::MIDDLE)
      return;
    //std::print("DRAG_END button {} from ({}, {}) to ({}, {})\n", static_cast<int>(button), pos0.x, pos0.y, pos.x, pos.y);
  };

  while (!workshop.shouldStop()) {
    workshop.beginFrame();
    
    winSize = workshop.getWindowSize();
    if (tex.specs.width != winSize.x || tex.specs.height != winSize.y) {
      if (texCuda) cudaGraphicsUnregisterResource(texCuda);
      tex.resize(winSize.x, winSize.y);
      cudaGraphicsGLRegisterImage(&texCuda, tex.getId(), GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore);
    }

    ImGui::Begin("Main");
    ImGui::Text("Frame No: %6d, Frame Dur: %.2f, FPS: %.1f", workshop.getFrameNo(), workshop.getFrameDurationMs(), workshop.getFrameRate());
    ImGui::Separator();
    static int maxIter = 100;
    static glm::vec3 bgColor{42 / 256.0, 96 / 256.0, 87 / 256.0};
    ImGui::ColorEdit3("BG Color", glm::value_ptr(bgColor));
    ImGui::SliderInt("Max Iteration", &maxIter, 1, 200);
    static bool useDouble = true;
    ImGui::Checkbox("Use Double", &useDouble);
    ImGui::Text("WinSize (%d, %d), TexSize (%d, %d)", winSize.x, winSize.y, tex.specs.width, tex.specs.height);
    ImGui::Text("Num Pixels: %d, Max Op: %d", winSize.x * winSize.y, winSize.x * winSize.y * maxIter);
    ImGui::DragFloat("x0", &model.topLeft.x, 0.001f);
    ImGui::DragFloat("y0", &model.topLeft.y, 0.001f);
    ImGui::DragFloat("h", &model.height, 0.001f);
    ImGui::Separator();
    static bool shouldShowImGuiDemo = false;
    ImGui::Checkbox("Show Demo", &shouldShowImGuiDemo);
    if (shouldShowImGuiDemo)
      ImGui::ShowDemoWindow();
    ImGui::End();

    cudaGraphicsMapResources(1, &texCuda, 0);
    cudaArray_t array;
    cudaGraphicsSubResourceGetMappedArray(&array, texCuda, 0, 0);
    struct cudaResourceDesc resDesc{};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = array;
    cudaSurfaceObject_t surface = 0;
    cudaCreateSurfaceObject(&surface, &resDesc);

    //launchGenSurface(surface, winSize.x, winSize.y, timeStep++);
    launchGenMandelbrot(surface, winSize.x, winSize.y, model.topLeft.x, model.topLeft.y, model.height, maxIter, useDouble, workshop.getFrameNo());

    cudaDestroySurfaceObject(surface);
    cudaGraphicsUnmapResources(1, &texCuda, 0);

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

  cudaGraphicsUnregisterResource(texCuda);

  return 0;
}