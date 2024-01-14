#include <Workshop/Shader.hpp>
#include <Workshop/Workshop.hpp>

#include <glad/gl.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <imgui.h>

#include <print>

int main()
{
  std::println("Hi!");
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

  // VAO binding is needed in 4.6 was not needed in 3.1
  uint32_t vao;
  glGenVertexArrays(1, &vao);

  while (!workshop.shouldStop()) {
    workshop.beginFrame();

    workshop.drawUI();

    ImGui::Begin("Triangle w/o VBO");
    static float bgColor[3] = {42 / 256.f, 96 / 256.f, 87 / 256.f};
    ImGui::ColorEdit3("BG Color", bgColor);
    ImGui::End();

    glBindVertexArray(vao);

    glClearColor(bgColor[0], bgColor[1], bgColor[2], 1);
    glClear(GL_COLOR_BUFFER_BIT);

    shader.bind();
    glDrawArrays(GL_TRIANGLES, 0, 3);

    workshop.endFrame();
  }

  std::println("Bye!");
  return 0;
}