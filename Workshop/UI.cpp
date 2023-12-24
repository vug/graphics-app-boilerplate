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

  ImGui::GetIO().ConfigWindowsMoveFromTitleBarOnly = true;
  ImGui::Begin("Texture Viewer", nullptr, ImGuiWindowFlags_NoScrollbar);
  auto items = textures 
    | std::views::transform([](const ws::Texture& tex) { return tex.getName().c_str(); }) 
    | std::ranges::to<std::vector<const char*>>();
  ImGui::Combo("Texture", &ix, items.data(), static_cast<uint32_t>(items.size()));
  const auto& tex = textures[ix].get();
  ImGui::Text("Name: %s, dim: (%d, %d)", tex.getName().c_str(), tex.specs.width, tex.specs.height);
  ImGui::Separator();

  const auto letterSize = ImGui::CalcTextSize("R");
  static std::array<bool, 4> channelSelection = {true, true, true, true};
  ImGui::Selectable("R", &channelSelection[0], ImGuiSelectableFlags_None, letterSize);
  ImGui::SameLine();
  ImGui::Selectable("G", &channelSelection[1], ImGuiSelectableFlags_None, letterSize);
  ImGui::SameLine();
  ImGui::Selectable("B", &channelSelection[2], ImGuiSelectableFlags_None, letterSize);
  ImGui::SameLine();
  ImGui::Selectable("A", &channelSelection[3], ImGuiSelectableFlags_None, letterSize);
  const ImVec4 tintColor = {static_cast<float>(channelSelection[0]), static_cast<float>(channelSelection[1]), static_cast<float>(channelSelection[2]), static_cast<float>(channelSelection[3])};

  static float zoomScale = 0.80f;
  ImGui::SliderFloat("scale", &zoomScale, 0.01f, 1.0f);

  const auto availableSize = ImGui::GetContentRegionAvail();
  const float availableAspectRatio = availableSize.x / availableSize.y;
  const float textureAspectRatio = static_cast<float>(tex.specs.width) / tex.specs.height;
  ImVec2 imgSize = (textureAspectRatio >= availableAspectRatio) ? ImVec2{availableSize.x, availableSize.x / textureAspectRatio} : ImVec2{availableSize.y * textureAspectRatio, availableSize.y};

  static ImVec2 uvOffset{0, 0};
  ImVec2 uvExtend{zoomScale, zoomScale};
  const ImVec2 drag = ImGui::GetMouseDragDelta(ImGuiMouseButton_Middle, 0);
  // one pixel of cursor move should pan the zoomed-in texture by one pixel
  ImVec2 deltaOffset{-drag.x / imgSize.x * zoomScale, drag.y / imgSize.y * zoomScale};

  // Only pan if drag operation starts while being hovered over the image
  static bool dragStartedFromImage = false;
  const ImVec2 pMin = ImGui::GetCursorScreenPos();
  const ImVec2 pMax = {pMin.x + imgSize.x, pMin.y + imgSize.y};
  const bool isHoveringOverImage = ImGui::IsMouseHoveringRect(pMin, pMax);
  if (ImGui::IsMouseClicked(ImGuiMouseButton_Middle))
    dragStartedFromImage = isHoveringOverImage;  

  ImVec2 off = uvOffset;
  if (dragStartedFromImage) {
    off.x += deltaOffset.x;
    off.y += deltaOffset.y;
    // only update uvOffset when dragging ends
    if (ImGui::IsMouseReleased(ImGuiMouseButton_Middle)) {
      uvOffset.x += deltaOffset.x;
      uvOffset.y += deltaOffset.y;
      deltaOffset.x = 0;
      deltaOffset.y = 0;
      dragStartedFromImage = false;
    }
  }
  // top-left corner (uv0) cannot be behind (0, 0) and bottom-right corner (uv1) cannot be beyond (1, 1)
  off.x = std::min(std::max(0.f, off.x), 1.f - uvExtend.x);
  off.y = std::min(std::max(0.f, off.y), 1.f - uvExtend.y);
  ImVec2 uv0 = {off.x, off.y};
  ImVec2 uv1 = {off.x + uvExtend.x, off.y + uvExtend.y};
  // flip the texture upside-down via following uv-coordinate transformation: (0, 0), (1, 1) -> (0, 1), (1, 0) 
  std::swap(uv0.y, uv1.y);
  
  ImGui::Image((void*)(intptr_t)tex.getId(), imgSize, uv0, uv1, tintColor, {0.5, 0.5, 0.5, 1.0});
  ImGui::End();
}


HierarchyWindow::HierarchyWindow(Scene& scene)
    : scene(scene) {}

VObjectPtr HierarchyWindow::draw() {
  ImGui::Begin("Hierarchy");

  struct NodeIdentifier {
    std::string name;
    int siblingId;
  };
  static NodeIdentifier selectedNodeId{"__NONE__", -1};
  static VObjectPtr selectedNode{};

  // ChildNo is relative to parent. It is used to give objects with the same name different ImGui ids
  std::function<void(VObjectPtr, int, int)> drawTree = [&](VObjectPtr node, int depth, int siblingId) {
    std::string nodeName = std::visit([](auto&& ptr) { return ptr->name; }, node);
    const auto& children = std::visit([](auto&& ptr) { return ptr->children; }, node);
    const bool hasChildren = !children.empty();
    ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_OpenOnArrow;  // | ImGuiTreeNodeFlags_SpanAvailWidth;
    if (!hasChildren)
      flags |= ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen;
    if (selectedNodeId.name == nodeName && selectedNodeId.siblingId == siblingId)
      flags |= ImGuiTreeNodeFlags_Selected;

    // TreeNode works is easier to use but puts the arrow to leaves
    const bool isOpen = ImGui::TreeNodeEx((void*)(intptr_t)siblingId, flags, nodeName.c_str());
    if (ImGui::IsItemClicked() && !ImGui::IsItemToggledOpen()) {
      selectedNodeId = {nodeName, siblingId};
      selectedNode = node;
    }
    if (isOpen) {
      int childNo = 1;
      for (auto childPtr : children)
        drawTree(childPtr, depth + 1, childNo++);

      if (hasChildren)
        ImGui::TreePop();
    }
  };

  drawTree(&scene.root, 0, 1);

  const bool isEmptyAreaClicked = ImGui::IsMouseClicked(ImGuiMouseButton_Left) && ImGui::IsWindowHovered() && !ImGui::IsAnyItemHovered();
  if (isEmptyAreaClicked) {
    selectedNodeId = {"__NONE__", -1};
    selectedNode = {};
  }

  ImGui::End();

  return selectedNode;
}
}