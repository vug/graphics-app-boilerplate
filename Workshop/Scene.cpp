#include "Scene.hpp"

namespace ws {
glm::mat4 Object::getLocalTransformMatrix() {
  return transform.getWorldFromObjectMatrix();
}

glm::mat4 Object::getGlobalTransformMatrix() {
  const bool hasParent = std::visit([](auto&& ptr) { return ptr != nullptr; }, parent);
  if (!hasParent)
    return getLocalTransformMatrix();

  return getLocalTransformMatrix() * std::visit([](auto&& ptr) { return ptr->getGlobalTransformMatrix(); }, parent);
}

void setParent(VObjectPtr child, VObjectPtr parent1) {
  std::visit([&child](auto&& ptr) { ptr->children.insert(child); }, parent1);
  std::visit([&parent1](auto&& ptr) { ptr->parent = parent1; }, child);
}

void traverse(ws::VObjectPtr node, int depth, NodeProcessor processNode) {
  const bool isNull = std::visit([](auto&& ptr) { return ptr == nullptr; }, node);
  if (isNull)
    return;

  processNode(node, depth);

  const auto& children = std::visit([](auto&& ptr) { return ptr->children; }, node);
  for (auto childPtr : children)
    traverse(childPtr, depth + 1, processNode);
}
}