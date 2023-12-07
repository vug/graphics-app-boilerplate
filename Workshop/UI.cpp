#include "UI.hpp"

#include <imgui/imgui.h>

#include <ranges>

namespace ws {
TextureViewer::TextureViewer(const std::vector<std::reference_wrapper<ws::Texture>>& textures) 
  : textures{textures} {
}

void TextureViewer::draw() {
  ImGui::Begin("Texture Viewer");
  drawCombo();
  drawTexture();
  ImGui::End();
}

void TextureViewer::drawCombo() {
  auto items = textures 
    | std::views::transform([](const ws::Texture& tex) { return tex.getName().c_str(); }) 
    | std::ranges::to<std::vector<const char*>>();
  ImGui::Combo("Texture", &ix, items.data(), 2);
}

void TextureViewer::drawTexture() {
  const auto& tex = textures[ix].get();
  ImGui::Text("Name: %s, dim: (%d, %d)", tex.getName().c_str(), tex.specs.width, tex.specs.height);
  ImGui::Image((void*)(intptr_t)tex.getId(), ImVec2{tex.specs.width / 4.f, tex.specs.height / 4.f}, {0, 1}, {1, 0});
}
}