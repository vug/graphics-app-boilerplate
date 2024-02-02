#include "UI.hpp"

#include <imgui/imgui.h>

namespace ws {
void ImGuiMaterialWidget(Material& mat) {
  ImGui::PushID(&mat);
  for (auto& [name, val] : mat.parameters) {
    std::visit(Overloaded{
      [&](int& val) { ImGui::DragInt(name.c_str(), &val); },
      [&](float& val) { ImGui::DragFloat(name.c_str(), &val);  },
      [&](glm::vec2& val) { ImGui::DragFloat2(name.c_str(), glm::value_ptr(val)); },
      [&](glm::vec3& val) { ImGui::DragFloat3(name.c_str(), glm::value_ptr(val)); },
    },
    val);
  }
  ImGui::PopID();
}
}