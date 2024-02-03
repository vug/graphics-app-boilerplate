#pragma once

#include "Common.hpp"
#include "Camera.hpp"
#include "Lights.hpp"
#include "Material.hpp"
#include "Model.hpp"
#include "Shader.hpp"
#include "Texture.hpp"
#include "Transform.hpp"
#include "UniformBuffer.hpp"

#include <functional>
#include <string>
#include <variant>
#include <unordered_set>

namespace ws {
struct RenderableObject;
struct RenderableObject2;
struct CameraObject;
struct DummyObject;
// using RenderableObjectRef = std::reference_wrapper<RenderableObject>;
// using CameraObjectRef = std::reference_wrapper<CameraObject>;
// using VObject = std::variant<RenderableObjectRef, CameraObjectRef>;
using VObjectPtr = std::variant<DummyObject*, RenderableObject*, RenderableObject2*, CameraObject*>;


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

struct RenderableObject2 : public Object {
  ws::Mesh& mesh;
  ws::Material& material;
};

struct CameraObject : public Object {
  ws::Camera camera;
};

// TODO: how can I move this into Object as a static member function
void setParent(VObjectPtr child, VObjectPtr parent1);

const int MAX_POINT_LIGHTS = 8;
const int MAX_DIRECTIONAL_LIGHTS = 4;

struct SceneUniforms {
  struct PaddedAmbientLight {
    glm::vec3 color;
    float _pad0{};

    PaddedAmbientLight() = default;
    PaddedAmbientLight(const AmbientLight& ambient);
  };
  struct PaddedHemisphericalLight {
    glm::vec3 northColor;
    float intensity;
    glm::vec3 southColor;
    float _pad0{};

    PaddedHemisphericalLight() = default;
    PaddedHemisphericalLight(const HemisphericalLight& hemispherical);
  };
  struct PaddedPointLight {
    glm::vec3 position;
    float intensity;
    glm::vec3 color;
    float _pad0{};

    PaddedPointLight() = default;
    PaddedPointLight(const PointLight& hemispherical);
  };
  struct PaddedDirectionalLight {
    glm::vec3 position;
    float intensity;
    //
    glm::vec3 direction;
    float _pad0{};
    //
    glm::vec3 color;
    float _pad1{};

    PaddedDirectionalLight() = default;
    PaddedDirectionalLight(const DirectionalLight& directional);
  };

  glm::mat4 u_ProjectionFromView;
  glm::mat4 u_ViewFromWorld;
  //
  glm::vec3 u_CameraPosition;
  float _pad0;
  //
  PaddedAmbientLight ambientLight;
  //
  PaddedHemisphericalLight hemisphericalLight;
  //
  glm::vec3 _pad1;
  int numPointLights;
  //
  PaddedPointLight pointLights[MAX_POINT_LIGHTS];
  //
  glm::vec3 _pad2;
  int numDirectionalLights;
  //
  PaddedDirectionalLight directionalLights[MAX_DIRECTIONAL_LIGHTS];
};

class Scene {
 public:
  DummyObject root{"SceneRoot", {glm::vec3{0, 0, 0}, glm::vec3{0, 1, 0}, 0, glm::vec3{1, 1, 1}}};
  std::vector<std::reference_wrapper<RenderableObject>> renderables;
  // Cameras
  Camera camera;
  // Lights
  AmbientLight ambientLight;
  HemisphericalLight hemisphericalLight;
  std::vector<PointLight> pointLights;
  std::vector<DirectionalLight> directionalLights;
  // not scene data
  UniformBuffer<SceneUniforms> ubo{1};
  void uploadUniforms();
};

using NodeProcessor = std::function<void(ws::VObjectPtr node, int depth)>;
void traverse(ws::VObjectPtr node, int depth, NodeProcessor processNode);
}