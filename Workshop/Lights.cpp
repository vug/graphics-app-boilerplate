#include "Lights.hpp"

#include <format>

namespace ws {

void AmbientLight::uploadToShader(const Shader& shader, const std::string& name) const {
  shader.setVector3(std::format("{}.color", name).c_str(), color);
}

void HemisphericalLight::uploadToShader(const Shader& shader, const std::string& name) const {
  shader.setVector3(std::format("{}.northColor", name).c_str(), northColor);
  shader.setFloat(std::format("{}.intensity", name).c_str(), intensity);
  shader.setVector3(std::format("{}.southColor", name).c_str(), southColor);
}

void PointLight::uploadToShader(const Shader& shader, const std::string& name) const {
  shader.setVector3(std::format("{}.position", name).c_str(), position);
  shader.setFloat(std::format("{}.intensity", name).c_str(), intensity);
  shader.setVector3(std::format("{}.color", name).c_str(), color);
}

void DirectionalLight::uploadToShader(const Shader& shader, const std::string& name) const {
  shader.setVector3(std::format("{}.position", name).c_str(), position);
  shader.setFloat(std::format("{}.intensity", name).c_str(), intensity);
  shader.setVector3(std::format("{}.direction", name).c_str(), direction);
  shader.setVector3(std::format("{}.color", name).c_str(), color);
}

} // namespace ws