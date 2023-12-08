#include "UI.hpp"

#include <imgui/imgui.h>

#include <algorithm>
#include <array>
#include <print>
#include <ranges>

namespace ws {

TextureViewer::TextureViewer(const std::vector<std::reference_wrapper<ws::Texture>>& textures) 
  : textures{textures} {
}

void TextureViewer::draw() {
  ImGui::Begin("Texture Viewer", nullptr, ImGuiWindowFlags_NoScrollbar);
  auto items = textures 
    | std::views::transform([](const ws::Texture& tex) { return tex.getName().c_str(); }) 
    | std::ranges::to<std::vector<const char*>>();
  ImGui::Combo("Texture", &ix, items.data(), 2);

  const auto letterSize = ImGui::CalcTextSize("R");
  static std::array<bool, 4> channelSelection = {true, true, true, true};
  ImGui::Selectable("R", &channelSelection[0], ImGuiSelectableFlags_None, letterSize);
  ImGui::SameLine();
  ImGui::Selectable("G", &channelSelection[1], ImGuiSelectableFlags_None, letterSize);
  ImGui::SameLine();
  ImGui::Selectable("B", &channelSelection[2], ImGuiSelectableFlags_None, letterSize);
  ImGui::SameLine();
  ImGui::Selectable("A", &channelSelection[3], ImGuiSelectableFlags_None, letterSize);
  const auto& tex = textures[ix].get();
  ImGui::Text("Name: %s, dim: (%d, %d)", tex.getName().c_str(), tex.specs.width, tex.specs.height);
  ImGui::Separator();

  const ImVec4 tintColor = {static_cast<float>(channelSelection[0]), static_cast<float>(channelSelection[1]), static_cast<float>(channelSelection[2]), static_cast<float>(channelSelection[3])};
  static float texScale = 0.80f;
  ImGui::SliderFloat("scale", &texScale, 0.01f, 1.0f);
  const auto availableSize = ImGui::GetContentRegionAvail();
  const float availableAspectRatio = availableSize.x / availableSize.y;
  const float textureAspectRatio = static_cast<float>(tex.specs.width) / tex.specs.height;
  ImVec2 size = (textureAspectRatio >= availableAspectRatio) ? ImVec2{availableSize.x, availableSize.x / textureAspectRatio} : ImVec2{availableSize.y * textureAspectRatio, availableSize.y};
  ImVec2 pos = ImGui::GetCursorScreenPos();
  static ImVec2 uvOffset{0, 0};
  ImVec2 uvExtend{1, 1};
  uvExtend.x *= texScale;
  uvExtend.y *= texScale;
  const ImVec2 drag = ImGui::GetMouseDragDelta(ImGuiMouseButton_Middle);
  ImVec2 deltaOffset {-drag.x / size.x, drag.y / size.y};
  if (ImGui::IsMouseReleased(ImGuiMouseButton_Middle)) {
    uvOffset.x += deltaOffset.x;
    uvOffset.y += deltaOffset.y;
    deltaOffset.x = 0;
    deltaOffset.y = 0;
  }
  ImVec2 off{uvOffset.x + deltaOffset.x, uvOffset.y + deltaOffset.y};
  off.x = std::min(std::max(0.f, off.x), 1.f - uvExtend.x);
  off.y = std::min(std::max(0.f, off.y), 1.f - uvExtend.y);
  // (0, 0), (1, 1) -> (0, 1), (1, 0)
  ImVec2 uv0 = {off.x, off.y + uvExtend.y};
  ImVec2 uv1 = {off.x + uvExtend.x, off.y};
  //ImVec2 uv0 = {off.x, off.y};
  //ImVec2 uv1 = {off.x + uvExtend.x, off.y + uvExtend.y};
  ImGui::Image((void*)(intptr_t)tex.getId(), size, uv0, uv1, tintColor, {0.5, 0.5, 0.5, 1.0});
  //ImGui::Image((void*)(intptr_t)tex.getId(), size, {0, texScale}, {texScale, 0}, tintColor, {0.5, 0.5, 0.5, 1.0});
  //ImGui::Image((void*)(intptr_t)tex.getId(), availableSize, {0, 1}, {1, 0}, tintColor, {0.5, 0.5, 0.5, 1.0});
  //ImGui::GetWindowDrawList()->AddRect({pos.x, pos.y}, {pos.x + availableSize.x, pos.y + availableSize.y}, IM_COL32(255, 0, 0, 255));
  ImGui::End();
}

}