#pragma once

#include <glm/vec3.hpp>

namespace ws {

struct AmbientLight {
  glm::vec3 color;
};

struct HemisphericalLight {
  glm::vec3 northColor;
  float intensity;
  glm::vec3 southColor;
};

struct PointLight {
  glm::vec3 position;
  float intensity;
  glm::vec3 color;
};

struct DirectionalLight {
  glm::vec3 position;
  float intensity;
  glm::vec3 direction;
  glm::vec3 color;
};
}