#pragma once

#include "Shader.hpp"

#include <glm/vec3.hpp>

namespace ws {

struct AmbientLight {
  glm::vec3 color;

  void uploadToShader(const Shader& shader, const std::string& name) const;
};

struct HemisphericalLight {
  glm::vec3 northColor;
  float intensity;
  glm::vec3 southColor;

  void uploadToShader(const Shader& shader, const std::string& name) const;
};

struct PointLight {
  glm::vec3 position;
  float intensity;
  glm::vec3 color;

  void uploadToShader(const Shader& shader, const std::string& name) const;
};

struct DirectionalLight {
  glm::vec3 position;
  float intensity;
  glm::vec3 direction;
  glm::vec3 color;

  void uploadToShader(const Shader& shader, const std::string& name) const;
};
}