#pragma once

#include "Common.hpp"
#include "Camera.hpp"
#include "Model.hpp"
#include "Shader.hpp"
#include "Texture.hpp"
#include "Transform.hpp"

#include <functional>
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

  glm::mat4 getLocalTransformMatrix();
  glm::mat4 getGlobalTransformMatrix();
};

struct DummyObject : public Object {};

struct RenderableObject : public Object {
  ws::Mesh& mesh;
  ws::Shader& shader;
  ws::Texture& texture;
  ws::Texture& texture2;
};

struct CameraObject : public Object {
  ws::Camera camera;
};

// TODO: how can I move this into Object as a static member function
void setParent(VObjectPtr child, VObjectPtr parent1);

class Scene {
 public:
  DummyObject root{"SceneRoot", {glm::vec3{0, 0, 0}, glm::vec3{0, 1, 0}, 0, glm::vec3{1, 1, 1}}};
  std::vector<std::reference_wrapper<RenderableObject>> renderables;
  // Cameras
  Camera camera;
};

using NodeProcessor = std::function<void(ws::VObjectPtr node, int depth)>;
void traverse(ws::VObjectPtr node, int depth, NodeProcessor processNode);
}