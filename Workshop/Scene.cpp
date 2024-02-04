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

SceneUniforms::PaddedAmbientLight::PaddedAmbientLight(const AmbientLight& ambient) : 
  color(ambient.color) 
{ }

SceneUniforms::PaddedHemisphericalLight::PaddedHemisphericalLight(const HemisphericalLight& hemispherical) :
  northColor(hemispherical.northColor),
  intensity(hemispherical.intensity),
  southColor(hemispherical.southColor)
{ }

SceneUniforms::PaddedPointLight::PaddedPointLight(const PointLight& point) : 
  position(point.position),
  intensity(point.intensity),
  color(point.color)
{ }

SceneUniforms::PaddedDirectionalLight::PaddedDirectionalLight(const DirectionalLight& directional) :
  position(directional.position),
  intensity(directional.intensity),
  direction(directional.direction),
  color(directional.color)
{ }

void Scene::uploadUniforms() {
  ubo.uniforms.u_ViewFromWorld = camera.getViewFromWorld();
  ubo.uniforms.u_ProjectionFromView = camera.getProjectionFromView();
  ubo.uniforms.u_CameraPosition = camera.position;
  // AmbientLight -> PaddedAmbientLight
  ubo.uniforms.ambientLight = ambientLight;
  // HemisphericalLight -> PaddedHemisphericalLight
  ubo.uniforms.hemisphericalLight = hemisphericalLight;
  // PointLight -> PaddedPointLight
  assert(pointLights.size() <= MAX_POINT_LIGHTS);
  ubo.uniforms.numPointLights = static_cast<int32_t>(pointLights.size());
  for (const auto& [ix, pl] : pointLights | std::ranges::views::enumerate)
    ubo.uniforms.pointLights[ix] = pl;
  // DirectionalLight -> PaddedDirectionalLight
  assert(directionalLights.size() <= MAX_DIRECTIONAL_LIGHTS);
  ubo.uniforms.numDirectionalLights = static_cast<int32_t>(directionalLights.size());
  for (const auto& [ix, dl] : directionalLights | std::ranges::views::enumerate)
    ubo.uniforms.directionalLights[ix] = dl;

  ubo.upload();
}

void Scene::draw() const {
  for (const auto& renderable : renderables | std::views::transform([](const auto& r) { return r.get(); })) {
    renderable.material.shader.bind();
    renderable.material.uploadParameters();
    renderable.material.shader.setMatrix4("u_WorldFromObject", renderable.transform.getWorldFromObjectMatrix());
    renderable.mesh.draw();
    renderable.material.shader.unbind();
  }
}

void Scene::draw(const Shader& overrideShader) const {
  for (const auto& renderable : renderables | std::views::transform([](const auto& r) { return r.get(); })) {
    overrideShader.bind();
    overrideShader.setMatrix4("u_WorldFromObject", renderable.transform.getWorldFromObjectMatrix());
    renderable.mesh.draw();
    overrideShader.unbind();
  }
}

void Scene::draw(const Material& overrideMaterial) const {
  for (const auto& renderable : renderables | std::views::transform([](const auto& r) { return r.get(); })) {
    overrideMaterial.shader.bind();
    overrideMaterial.uploadParameters();
    overrideMaterial.shader.setMatrix4("u_WorldFromObject", renderable.transform.getWorldFromObjectMatrix());
    renderable.mesh.draw();
    overrideMaterial.shader.unbind();
  }
}
}