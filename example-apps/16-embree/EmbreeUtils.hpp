#pragma once

#include <embree4/rtcore.h>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <filesystem>

namespace eu {

struct ERayResult {
  RTCScene scene;
  glm::vec3 origin;
  glm::vec3 direction;
  glm::vec3 position;
  glm::vec3 faceNormal;
  glm::vec2 faceUv;
  bool hasHit;
  bool hasMissed;
  uint32_t geomId{RTC_INVALID_GEOMETRY_ID};
  uint32_t primId{RTC_INVALID_GEOMETRY_ID};
  RTCGeometry geom;

  template <typename TVecN>
  TVecN interpolateVertexAttribute(int bufferSlot) const {
    TVecN result;
    rtcInterpolate0(geom, primId, faceUv.x, faceUv.y, RTC_BUFFER_TYPE_VERTEX_ATTRIBUTE, bufferSlot, glm::value_ptr(result), result.length());
    return result;
  };
};

class ERay {
 public:
  ERay(RTCScene scene);
  ERay(RTCScene scene, const glm::vec3& o, const glm::vec3& d);

  const ERayResult intersect();

  // No need to use the methods below, instead prefer using a const ERayResult
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

glm::vec2 dirToLonLat(glm::vec3 dir);

float rndUni01();

glm::vec3 sampleLambertian(const glm::vec3& norm, const glm::vec3& in, float spread = 1.f);

RTCGeometry makeTriangularGeometry(RTCDevice dev, const std::vector<glm::vec3>& verts, const std::vector<glm::vec3>& norms, const std::vector<glm::vec2>& texCoords, const std::vector<uint32_t>& ixs);

class Image {
 public:
  Image(const std::filesystem::path& path);
  glm::vec3 nearest(float x, float y) const;
  int getWidth() const { return width; }
  int getHeight() const { return height; }
  int getNumChannels() const { return numChannels; }

 private:
  int width;
  int height;
  int numChannels;
  std::vector<glm::vec3> pixels;
};
}  // namespace ws
