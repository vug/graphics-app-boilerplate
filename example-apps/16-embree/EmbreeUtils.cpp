#include "EmbreeUtils.hpp"

namespace ws {

ERay::ERay(RTCScene scene)
    : ERay(scene, glm::vec3(0), glm::vec3(0, 0, 1)) {
}

ERay::ERay(RTCScene scene, const glm::vec3& o, const glm::vec3& d)
    : scene_{scene},
      rh_{
          .ray = {
              .org_x = o.x,
              .org_y = o.y,
              .org_z = o.z,
              .tnear = 0.0001f,
              .dir_x = d.x,
              .dir_y = d.y,
              .dir_z = d.z,
              .tfar = std::numeric_limits<float>::infinity(),
              .mask = static_cast<unsigned>(-1),
          },
          .hit{
              .geomID = RTC_INVALID_GEOMETRY_ID,
              .instID = {RTC_INVALID_GEOMETRY_ID},
          }},
      ori_{reinterpret_cast<glm::vec3*>(&rh_.ray.org_x)},
      dir_{reinterpret_cast<glm::vec3*>(&rh_.ray.dir_x)},
      norm_{reinterpret_cast<glm::vec3*>(&rh_.hit.Ng_x)},
      uv_{reinterpret_cast<glm::vec2*>(&rh_.hit.u)} {
}

const ERayResult ERay::intersect() {
  rtcIntersect1(scene_, &rh_);
  ERayResult result{
    .scene=scene_,
    .origin=*ori_,
    .direction=*dir_,
    .position=*ori_ + *dir_ * rh_.ray.tfar,
    .faceNormal=*norm_,
    .faceUv=*uv_,
    .hasHit=rh_.hit.geomID != RTC_INVALID_GEOMETRY_ID,
    .hasMissed=rh_.hit.geomID == RTC_INVALID_GEOMETRY_ID,
    .geomId=rh_.hit.geomID,
    .primId=rh_.hit.primID,
    .geom=hasHit() ? rtcGetGeometry(scene_, rh_.hit.geomID) : nullptr,
  };

  return result;
}
}  // namespace ws