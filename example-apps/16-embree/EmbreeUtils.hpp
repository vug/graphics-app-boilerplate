#pragma once

#include <embree4/rtcore.h>
#include <glm/glm.hpp>

namespace ws {
class ERay {
 public:
  ERay(RTCScene scene);
  ERay(RTCScene scene, const glm::vec3& o, const glm::vec3& d);

  void intersect();

  inline glm::vec3& direction() const { return *dir_; }
  inline glm::vec3& origin() const { return *ori_; }
  inline glm::vec3& normal() const { return *norm_; }
  inline glm::vec2& uv() const { return *uv_; }
  inline bool hasHit() const { return geomId() != RTC_INVALID_GEOMETRY_ID; }
  inline bool hasMissed() const { return geomId() == RTC_INVALID_GEOMETRY_ID; }
  inline uint32_t geomId() const { return rh_.hit.geomID; }
  inline uint32_t primId() const { return rh_.hit.primID; }
  inline glm::vec3 pos() const { return origin() + direction() * rh_.ray.tfar; }

 private:
  RTCScene scene_{};
  RTCRayHit rh_{};
  glm::vec3* dir_{};
  glm::vec3* ori_{};
  glm::vec3* norm_{};
  glm::vec2* uv_{};
};
}  // namespace ws
