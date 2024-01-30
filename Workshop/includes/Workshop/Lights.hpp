#pragma once

#include <glm/vec3.hpp>

namespace ws {

struct AmbientLight {
  glm::vec3 color;
  float _pad0;
};

struct HemisphericalLight {
  glm::vec3 northColor;
  float intensity;
  glm::vec3 southColor;
  float _pad0;
};

struct PointLight {
  glm::vec3 position;
  float intensity;
  glm::vec3 color;
  float _pad0;
};

const int MAX_DIRECTIONAL_LIGHTS = 4;
struct DirectionalLight {
  glm::vec3 position;
  float intensity;
  //
  glm::vec3 direction;
  float _pad0;
  //
  glm::vec3 color;
  float _pad1;
};
}