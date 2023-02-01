#include "Workshop.h"

#include <glad/gl.h>
#include <imgui.h>

#include <stb_image.h>
#include <tiny_obj_loader.h>
#include <vivid/vivid.h>
#include <glm/gtc/type_ptr.hpp>
#include <glm/vec3.hpp>

#include <iostream>

int main() {
  std::cout << "Hello, RenderGraph!\n";
  ws::Workshop workshop{800, 600, "Workshop App"};

  while (!workshop.shouldStop()) {
    workshop.beginFrame();

    ImGui::Begin("Main");
    static bool shouldShowImGuiDemo = false;
    ImGui::Checkbox("Show Demo", &shouldShowImGuiDemo);
    ImGui::SetNextWindowPos(ImVec2{5, 5});
    if (shouldShowImGuiDemo)
      ImGui::ShowDemoWindow();

    static glm::vec3 bgColor{1, 0, 0};
    ImGui::ColorEdit3("BG Color", glm::value_ptr(bgColor));
    ImGui::End();

    glClearColor(bgColor.x, bgColor.y, bgColor.z, 1);
    glClear(GL_COLOR_BUFFER_BIT);

    workshop.endFrame();
  }

  return 0;
}