#include <Workshop/Workshop.hpp>

#include <glad/gl.h>
#include <imgui.h>

int main() {
  ws::Workshop workshop{800, 600, "Workshop App"};

  while (!workshop.shouldStop()) {
    workshop.beginFrame();

    workshop.drawUI();

    ImGui::Begin("Clear");
    static glm::vec4 bgColor{1, 0, 0, 1};
    ImGui::ColorEdit4("BG Color", glm::value_ptr(bgColor));
    ImGui::End();

    glClearNamedFramebufferfv(0, GL_COLOR, 0, glm::value_ptr(bgColor));

    workshop.endFrame();
  }

  return 0;
}