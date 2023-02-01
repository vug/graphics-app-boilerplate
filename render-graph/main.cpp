#include "Workshop.h"

#include <glad/gl.h>

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <implot.h>
#include <stb_image.h>
#include <tiny_obj_loader.h>
#include <vivid/vivid.h>
#include <glm/vec3.hpp>

#include <iostream>

int main() {
  std::cout << "Hello, RenderGraph!\n";
  ws::Workshop workshop{};

  while (!workshop.shouldStop()) {
    workshop.beginFrame();

    glClearColor(1, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);

    workshop.endFrame();
  }

  return 0;
}