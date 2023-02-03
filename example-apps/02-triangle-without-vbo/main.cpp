#include <Shader.h>
#include <Workshop.h>

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

int main()
{
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

    glClearColor(bgColor.x, bgColor.y, bgColor.z, 1);
    glClear(GL_COLOR_BUFFER_BIT);

    shader.bind();
    glDrawArrays(GL_TRIANGLES, 0, 3);

    workshop.endFrame();
  }

  return 0;
}