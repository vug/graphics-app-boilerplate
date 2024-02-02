#include "UI.hpp"

#include <imgui/imgui.h>

#include <ranges>

namespace ws {
void ImGuiMaterialWidget(Material& mat, AssetManager& assetManager) {
  ImGui::PushID(&mat);
  for (auto& [name, val] : mat.parameters) {
    std::visit(Overloaded{
      [&](int& val) { ImGui::DragInt(name.c_str(), &val); },
      [&](float& val) { ImGui::DragFloat(name.c_str(), &val);  },
      [&](glm::vec2& val) { ImGui::DragFloat2(name.c_str(), glm::value_ptr(val)); },
      [&](glm::vec3& val) { ImGui::DragFloat3(name.c_str(), glm::value_ptr(val)); },
      [&](Texture& val) { 
        const auto texNames = assetManager.textures | std::views::transform([](auto& items) { return items.second.getName(); }) | std::ranges::to<std::vector>();
        const auto texLabels = assetManager.textures | std::views::transform([](auto& items) { return items.first; }) | std::ranges::to<std::vector>();
        // Combo works with const char*
        const auto texNamesCstr = texNames | std::views::transform([](auto& str) { return str.c_str(); }) | std::ranges::to<std::vector>();
        // Get currently chosen texture's name's index in texNames vector
        auto it = std::ranges::find(texNames, val.getName());
        assert(it != texNames.end());
        int32_t vectorIx = std::distance(texNames.begin(), it);
        // control vectorIx via UI, and if updated, set material texture parameter to the newly chosen one
        if (ImGui::Combo("Texture", &vectorIx, texNamesCstr.data(), static_cast<uint32_t>(texNamesCstr.size()))) {
          std::string chosenTexName = texLabels[vectorIx];
          mat.parameters.at(name) = assetManager.textures[chosenTexName];
        }
      },
    },
    val);
  }
  ImGui::PopID();
}
}