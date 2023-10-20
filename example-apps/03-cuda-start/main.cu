// Main file had to be main.cu and not main.cpp otherwise <<< is not recognized.
// I was able to overcome this in VS Code by compiling CU files via NVCC into a library and compile main.cpp via MSVC and link them later
#include "vector_add.h"

#include <Workshop/Shader.hpp>
#include <Workshop/Workshop.hpp>

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

int main(int argc, char* argv[]) {
  const int count = (argc == 2) ? std::stoi(argv[1]) : 5;
  const size_t size = count * sizeof(float);

  // Allocate memory of inputs on host, fill them out
  std::vector<float> h_a(count), h_b(count), h_c(count);
  for (int ix = 0; ix < count; ++ix) {
    h_a[ix] = ix;
    h_b[ix] = 2 * ix;
  }

  // Allocate memory for device counterparts
  float *d_a, *d_b, *d_c;
  cudaMalloc(&d_a, size);
  cudaMalloc(&d_b, size);
  cudaMalloc(&d_c, size);

  // Copy inputs from host to device
  cudaMemcpy(d_a, h_a.data(), size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b.data(), size, cudaMemcpyHostToDevice);

  // Run kernel
  vectorAdd<<<1, size>>>(d_a, d_b, d_c);

  // Copy the result back to host
  cudaMemcpy(h_c.data(), d_c, size, cudaMemcpyDeviceToHost);

  // "Use the result"
  for (int i = 0; i < count; ++i)
    std::cout << std::format("{} + {} = {}\n", h_a[i], h_b[i], h_c[i]);

  // Free memory
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  std::cout << "Hi!\n";
  ws::Workshop workshop{800, 600, "Workshop App"};

  const char *vertexShader = R"(
#version 300 es
#extension GL_EXT_separate_shader_objects : enable
precision mediump float;

layout (location = 0) out vec3 fragColor;

vec2 positions[3] = vec2[](vec2 (0.0, -0.5), vec2 (0.5, 0.5), vec2 (-0.5, 0.5));
vec3 colors[3] = vec3[](vec3 (1.0, 0.0, 0.0), vec3 (0.0, 1.0, 0.0), vec3 (0.0, 0.0, 1.0));
void main ()
{
	gl_Position = vec4 (positions[gl_VertexID], 0.0, 1.0);
	fragColor = colors[gl_VertexID];
}
  )";

  const char *fragmentShader = R"(
#version 300 es
#extension GL_EXT_separate_shader_objects : enable
precision mediump float;

layout (location = 0) in vec3 fragColor;
layout (location = 0) out vec4 outColor;

void main () { outColor = vec4 (fragColor, 1.0); }
  )";
  ws::Shader shader{vertexShader, fragmentShader};

  while (!workshop.shouldStop())
  {
    workshop.beginFrame();

    ImGui::Begin("Main");
    static bool shouldShowImGuiDemo = false;
    ImGui::Checkbox("Show Demo", &shouldShowImGuiDemo);
    if (shouldShowImGuiDemo)
      ImGui::ShowDemoWindow();

    static glm::vec3 bgColor{42 / 256.0, 96 / 256.0, 87 / 256.0};
    ImGui::ColorEdit3("BG Color", glm::value_ptr(bgColor));
    ImGui::End();

    // VAO binding is needed in 4.6 was not needed in 3.1
    uint32_t vao;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    glClearColor(bgColor.x, bgColor.y, bgColor.z, 1);
    glClear(GL_COLOR_BUFFER_BIT);

    shader.bind();
    glDrawArrays(GL_TRIANGLES, 0, 3);

    workshop.endFrame();
  }

  return 0;
}