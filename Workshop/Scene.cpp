#include "Scene.hpp"

#include <ranges>

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
  if (isNull) {
    assert(false); // shouldn't encounter an null node while traversing
    return;
  }

  processNode(node, depth);

  const auto& children = std::visit([](auto&& ptr) { return ptr->children; }, node);
  for (auto childPtr : children)
    traverse(childPtr, depth + 1, processNode);
}

void Scene::uploadUniforms() {
  ubo.uniforms.u_ViewFromWorld = camera.getViewFromWorld();
  ubo.uniforms.u_ProjectionFromView = camera.getProjectionFromView();
  ubo.uniforms.u_CameraPosition = camera.position;
  // AmbientLight -> PaddedAmbientLight
  ubo.uniforms.ambientLight.color = ambientLight.color;
  // HemisphericalLight -> PaddedHemisphericalLight
  ubo.uniforms.hemisphericalLight.northColor = hemisphericalLight.northColor;
  ubo.uniforms.hemisphericalLight.intensity = hemisphericalLight.intensity;
  ubo.uniforms.hemisphericalLight.southColor = hemisphericalLight.southColor;
  assert(pointLights.size() <= MAX_POINT_LIGHTS);
  // PointLight -> PaddedPointLight
  ubo.uniforms.numPointLights = static_cast<int32_t>(pointLights.size());
  for (const auto& [ix, pl] : pointLights | std::ranges::views::enumerate) {
    ubo.uniforms.pointLights[ix].position = pl.position;
    ubo.uniforms.pointLights[ix].intensity = pl.intensity;
    ubo.uniforms.pointLights[ix].color = pl.color;
  }
  // DirectionalLight -> PaddedDirectionalLight
  assert(directionalLights.size() <= MAX_DIRECTIONAL_LIGHTS);
  ubo.uniforms.numDirectionalLights = static_cast<int32_t>(directionalLights.size());
  for (const auto& [ix, dl] : directionalLights | std::ranges::views::enumerate) {
    ubo.uniforms.directionalLights[ix].position = dl.position;
    ubo.uniforms.directionalLights[ix].intensity = dl.intensity;
    ubo.uniforms.directionalLights[ix].direction = dl.direction;
    ubo.uniforms.directionalLights[ix].color = dl.color;
  }

  ubo.upload();
}
}