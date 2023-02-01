#include "Workshop.h"

#include <glad/gl.h>

#include <stb_image.h>
#include <tiny_obj_loader.h>
#include <vivid/vivid.h>
#include <glm/vec3.hpp>

#include <iostream>

int main() {
  std::cout << "Hello, RenderGraph!\n";
  ws::Workshop workshop{800, 600, "Workshop App"};

  while (!workshop.shouldStop()) {
    workshop.beginFrame();

    glClearColor(1, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);

    workshop.endFrame();
  }

  return 0;
}