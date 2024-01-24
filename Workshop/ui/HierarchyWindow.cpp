#include "ui/HierarchyWindow.hpp"

#include <imgui/imgui.h>

namespace ws {
HierarchyWindow::HierarchyWindow(Scene& scene)
    : scene(scene) {}

VObjectPtr HierarchyWindow::draw(VObjectPtr clickedObject) {
  ImGui::Begin("Hierarchy");

  struct NodeIdentifier {
    std::string name;
    int siblingId;
  };
  static NodeIdentifier selectedNodeId{"__NONE__", -1};
  static VObjectPtr selectedNode{};

  std::function<void(VObjectPtr, int, int)> findClickedInTree = [&](VObjectPtr node, int depth, int siblingId) {
    if (node == clickedObject) {
      std::string nodeName = std::visit([](auto&& ptr) { return ptr->name; }, node);
      selectedNodeId = {nodeName, siblingId};
      selectedNode = node;
    }
    const auto& children = std::visit([](auto&& ptr) { return ptr->children; }, node);
    int childNo = 1;
    for (auto childPtr : children)
      findClickedInTree(childPtr, depth + 1, childNo++);
  };
  findClickedInTree(&scene.root, 0, 1);

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

    // TreeNode is easier to use but always puts an arrow to leaf nodes
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