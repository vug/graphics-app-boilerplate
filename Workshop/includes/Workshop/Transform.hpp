#pragma once

#include <glm/fwd.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/vec3.hpp>

namespace ws {
class Transform {
 public:
  glm::vec3 position;
  glm::quat rotation;
  glm::vec3 scale;

  Transform(const glm::vec3& pos, const glm::quat& rot, const glm::vec3& sca);
  Transform(const glm::vec3& pos, const glm::vec3 axis, const float angle, const glm::vec3& sca);

  glm::mat4 getTranslateMatrix() const;
  glm::mat4 getRotationMatrix() const;
  glm::mat4 getScaleMatrix() const;
  glm::mat4 getWorldFromObjectMatrix() const;
};

glm::quat rotateTowards(glm::quat q1, glm::quat q2, float maxAngle);
glm::mat4 removeTranslation(const glm::mat4 mat);
}