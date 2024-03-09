#include "EmbreeUtils.hpp"

#include <stb_image.h>

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

Image::Image(const std::filesystem::path& path) {
  const auto ext = path.extension();
  const bool isHdr = ext == "hdr";
  if (!isHdr) {
  uint8_t* data = stbi_load(path.string().c_str(), &width, &height, &numChannels, 0);
  assert(numChannels == 3);
    pixels.resize(width * height);
  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
      const int ix = i * width + j;
      pixels[ix] = {data[3 * ix + 0], data[3 * ix + 1], data[3 * ix + 2]};
      pixels[ix] /= 255.99;
    }
  }
  delete data;
  } else {
    float* data = stbi_loadf(path.string().c_str(), &width, &height, &numChannels, 0);
    pixels.resize(width * height);
    for (int i = 0; i < height; ++i) {
      for (int j = 0; j < width; ++j) {
        const int ix = i * width + j;
        pixels[ix] = {data[3 * ix + 0], data[3 * ix + 1], data[3 * ix + 2]};
      }
}
    delete data;
  }
}

glm::vec3 Image::nearest(float x, float y) const {
  x = std::max(std::min(x, width - 1.f), 0.f); // clamp
  y = std::max(std::min(y, height - 1.f), 0.f);
  int i = std::floor(x);
  int j = std::floor(y);
  glm::vec3 result = pixels[i * width + j];
  return result;
}
}  // namespace ws