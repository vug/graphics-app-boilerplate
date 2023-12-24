#include "UI.hpp"

#include <glm/gtx/euler_angles.hpp>
#include <imgui/imgui.h>
#include <imgui_internal.h> // for PushMultiItemsWidths, GImGui

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

// taken from a The Cherno tutorial
static bool DrawVec3Control(const std::string& label, glm::vec3& values, float resetValue = 0.0f, float columnWidth = 100.0f) {
  bool value_changed = false;
  ImGui::PushID(label.c_str());

  ImGui::Columns(2);
  ImGui::SetColumnWidth(0, columnWidth);
  ImGui::Text(label.c_str());
  ImGui::NextColumn();

  ImGui::PushMultiItemsWidths(3, ImGui::CalcItemWidth());
  ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2{0, 0});

  float lineHeight = GImGui->Font->FontSize + GImGui->Style.FramePadding.y * 2.0f;
  ImVec2 buttonSize = {lineHeight + 3.0f, lineHeight};

  // https://coolors.co/ff595e-ffca3a-8ac926-1982c4-6a4c93
  ImGui::PushStyleColor(ImGuiCol_Button, ImVec4{1.0f, 0.34901961f, 0.36862745f, 1.0f});
  ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4{1.000000f, 0.521569f, 0.537255f, 1.0f});
  ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4{1.000000f, 0.200000f, 0.227451f, 1.0f});

  if (ImGui::Button("X", buttonSize)) {
    values.x = resetValue;
    value_changed = true;
  }
  ImGui::PopStyleColor(3);

  ImGui::SameLine();
  value_changed |= ImGui::DragFloat("##X", &values.x, 0.1f, 0.0f, 0.0f, "%.2f");
  ImGui::PopItemWidth();
  ImGui::SameLine();

  ImGui::PushStyleColor(ImGuiCol_Button, ImVec4{0.541176f, 0.788235f, 0.149020f, 1.0f});
  ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4{0.631373f, 0.858824f, 0.262745f, 1.0f});
  ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4{0.462745f, 0.670588f, 0.129412f, 1.0f});
  if (ImGui::Button("Y", buttonSize)) {
    values.y = resetValue;
    value_changed = true;
  }
  ImGui::PopStyleColor(3);

  ImGui::SameLine();
  value_changed |= ImGui::DragFloat("##Y", &values.y, 0.1f, 0.0f, 0.0f, "%.2f");
  ImGui::PopItemWidth();
  ImGui::SameLine();

  ImGui::PushStyleColor(ImGuiCol_Button, ImVec4{0.098039f, 0.509804f, 0.768627f, 1.0f});
  ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4{0.149020f, 0.607843f, 0.890196f, 1.0f});
  ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4{0.082353f, 0.423529f, 0.635294f, 1.0f});
  if (ImGui::Button("Z", buttonSize)) {
    values.z = resetValue;
    value_changed = true;
  }
  ImGui::PopStyleColor(3);

  ImGui::SameLine();
  value_changed |= ImGui::DragFloat("##Z", &values.z, 0.1f, 0.0f, 0.0f, "%.2f");
  ImGui::PopItemWidth();

  ImGui::PopStyleVar();
  ImGui::Columns(1);
  ImGui::PopID();

  return value_changed;
}

void InspectorWindow::inspectObject(VObjectPtr objPtr) {
  ImGui::Begin("Inspector");

  const bool isNull = std::visit([](auto&& objPtr) { return objPtr == nullptr; }, objPtr);
  if (isNull) {
    ImGui::Text("Nothing Selected");
    ImGui::End();
    return;
  }

  const std::string& name = std::visit([](auto&& objPtr) { return objPtr->name; }, objPtr);
  ImGui::Text("%s", name.c_str());

  Transform& transform = std::visit([](auto&& objPtr) -> Transform& { return objPtr->transform; }, objPtr);
  DrawVec3Control("Position", transform.position);
  glm::vec3 eulerXyzDeg = glm::degrees(glm::eulerAngles(transform.rotation));
  if (DrawVec3Control("Rotation", eulerXyzDeg))
    transform.rotation = glm::quat(glm::radians(eulerXyzDeg));
  DrawVec3Control("Scale", transform.scale, 1);

  ImGui::End();
}
}