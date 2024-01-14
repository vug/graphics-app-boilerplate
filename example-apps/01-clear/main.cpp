#include <Workshop/Workshop.hpp>

#include <glad/gl.h>
#include <imgui.h>

int main() {
  ws::Workshop workshop{800, 600, "Workshop App"};

  while (!workshop.shouldStop()) {
    workshop.beginFrame();

    workshop.drawUI();

    ImGui::Begin("Main");
    static float bgColor[3] = {1.f, 0.f, 0.f};
    ImGui::ColorEdit3("BG Color", bgColor);
    ImGui::End();

    glClearColor(bgColor[0], bgColor[1], bgColor[2], 1);
    glClear(GL_COLOR_BUFFER_BIT);

    workshop.endFrame();
  }

  return 0;
}