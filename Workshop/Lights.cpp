#include "Lights.hpp"

#include <fmt/core.h>
#include <glm/gtc/type_ptr.hpp>

namespace ws {

void PointLight::uploadToShader(const Shader& shader, int ix) const {
  shader.setVector3(fmt::format("pointLights[{}].color", ix).c_str(), color);
  shader.setVector3(fmt::format("pointLights[{}].position", ix).c_str(), position);
  shader.setFloat(fmt::format("pointLights[{}].intensity", ix).c_str(), intensity);
}

void DirectionalLight::uploadToShader(const Shader& shader, int ix) const {
  shader.setVector3(fmt::format("directionalLights[{}].color", ix).c_str(), color);
  shader.setVector3(fmt::format("directionalLights[{}].direction", ix).c_str(), direction);
  shader.setFloat(fmt::format("directionalLights[{}].intensity", ix).c_str(), intensity);
}

void HemisphericalLight::uploadToShader(const Shader& shader) const {
  shader.setVector3(fmt::format("hemisphericalLight.northColor").c_str(), northColor);
  shader.setVector3(fmt::format("hemisphericalLight.southColor").c_str(), southColor);
  shader.setFloat(fmt::format("hemisphericalLight.intensity").c_str(), intensity);
}

void AmbientLight::uploadToShader(const Shader& shader) const {
  shader.setVector3(fmt::format("ambientLight.color").c_str(), color);
}

} // namespace ws