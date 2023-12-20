#pragma once

#include "Common.hpp"
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
  std::vector<std::reference_wrapper<RenderableObject>> renderables;
  std::vector<std::reference_wrapper<CameraObject>> cameras;
};

template <class... Ts>
void traverse(ws::VObjectPtr node, int depth, Overloaded<Ts...> ovr) {
  const bool isNull = std::visit([](auto&& ptr) { return ptr == nullptr; }, node);
  if (isNull)
    return;

  //ws::VObjectPtr parentNode = std::visit([](auto&& ptr) { return ptr->parent; }, node);
  //const std::string& parentName = std::visit([](auto&& ptr) { return ptr != nullptr ? ptr->name : "NO_PARENT"; }, parentNode);

  std::visit(ovr, node);

  const auto& children = std::visit([](auto&& ptr) { return ptr->children; }, node);
  for (auto childPtr : children)
    traverse(childPtr, depth + 1, ovr);
}

// Not abstracted traverse that can utilize depth and parentNode
//void traverse(ws::VObjectPtr node, int depth) {
//  const bool isNull = std::visit([](auto&& ptr) { return ptr == nullptr; }, node);
//  if (isNull)
//    return;
//
//  ws::VObjectPtr parentNode = std::visit([](auto&& ptr) { return ptr->parent; }, node);
//  const std::string& parentName = std::visit([](auto&& ptr) { return ptr != nullptr ? ptr->name : "NO_PARENT"; }, parentNode);
//
//  std::visit(Overloaded{
//                 [](auto arg) { throw "Unhandled VObjectPtr variant"; },
//                 [&](ws::DummyObject* ptr) {
//                   std::println("{} parent {} DummyObject name {}", depth, parentName, ptr->name);
//                 },
//                 [&](ws::RenderableObject* ptr) {
//                   ws::RenderableObject& ref = *ptr;
//                   std::println("{} parent {} RenderableObject name {} verts {} tr.pos.x {}", depth, parentName, ref.name, ref.mesh.meshData.vertices.size(), ref.transform.position.x);
//                 },
//                 [&](ws::CameraObject* ptr) {
//                   ws::CameraObject& ref = *ptr;
//                   std::println("{} parent {} CameraObject name {} verts fov {} tr.pos.x {}", depth, parentName, ref.name, ref.camera.fov, ref.transform.position.x);
//                 },
//             },
//             node);
//
//  const auto& children = std::visit([](auto&& ptr) { return ptr->children; }, node);
//  for (auto childPtr : children)
//    traverse(childPtr, depth + 1);
//}
}