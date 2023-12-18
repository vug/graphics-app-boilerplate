#pragma once

#include "Camera.hpp"
#include "Model.hpp"
#include "Shader.hpp"
#include "Texture.hpp"
#include "Transform.hpp"

#include <string>
#include <variant>
#include <unordered_set>

namespace ws {
struct RenderableObject;
struct CameraObject;
struct DummyObject;
// using RenderableObjectRef = std::reference_wrapper<RenderableObject>;
// using CameraObjectRef = std::reference_wrapper<CameraObject>;
// using VObject = std::variant<RenderableObjectRef, CameraObjectRef>;
using VObjectPtr = std::variant<DummyObject*, RenderableObject*, CameraObject*>;


struct Object {
  std::string name;
  ws::Transform transform;

  VObjectPtr parent;
  std::unordered_set<VObjectPtr> children;
};

struct DummyObject : public Object {};

struct RenderableObject : public Object {
  ws::Mesh& mesh;
  ws::Shader& shader;
  ws::Texture& texture;
};

struct CameraObject : public Object {
  ws::PerspectiveCamera3D camera;
};

// TODO: how can I move this into Object as a static member function
void setParent(VObjectPtr child, VObjectPtr parent1) {
  std::visit([&child](auto&& ptr) { ptr->children.insert(child); }, parent1);
  std::visit([&parent1](auto&& ptr) { ptr->parent = parent1; }, child);
}

class Scene {
 public:
  DummyObject root{"SceneRoot", {glm::vec3{0, 0, 0}, glm::vec3{0, 1, 0}, 0, glm::vec3{1, 1, 1}}};
  std::vector<RenderableObject> renderables;
  std::vector<CameraObject> cameras;
};
}