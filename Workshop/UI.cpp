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
        const auto texNames = assetManager.textures | std::views::values | std::views::transform([](auto& tex) { return tex.getName(); }) | std::ranges::to<std::vector>();
        const auto texLabels = assetManager.textures | std::views::keys | std::ranges::to<std::vector>();
        // Combo works with const char* but for some reason putting a c_str() after format() does not work.
        // It complains about "pointer dangling because it points at a temporary instance that was destroyed".
        // Therefore I need to first store them in std::vector<string> texLabelNames then convert that to c_str() in a next pass.
        const auto texLabelNames = std::views::zip_transform([](auto& a, auto& b) { return std::format("{}/{}", a, b); }, texLabels, texNames) | std::ranges::to<std::vector>();
        const auto texLabelNamesCstr = texLabelNames | std::views::transform([](auto& str) { return str.c_str(); }) | std::ranges::to<std::vector>();
        // Get currently the index of the chosen texture's name in texNames vector
        auto it = std::ranges::find(texNames, val.getName());
        assert(it != texNames.end());
        int32_t vectorIx = std::distance(texNames.begin(), it);
        // control vectorIx via UI, and if updated, set material texture parameter to the newly chosen one
        if (ImGui::Combo("Texture", &vectorIx, texLabelNamesCstr.data(), static_cast<uint32_t>(texLabelNamesCstr.size()))) {
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