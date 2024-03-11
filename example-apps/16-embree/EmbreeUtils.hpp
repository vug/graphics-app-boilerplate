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

class ToneMapper {
 public:
  static glm::vec3 toGamma(const glm::vec3& rgb) {
    return glm::pow(abs(rgb), glm::vec3(1.0f / 2.2f));
  }
  static glm::vec3 toReinhard1(const glm::vec3& rgb) {
    float lumRgb = lumf(rgb);
    float lumScale = lumRgb / (lumRgb + 1.0f);
    return toGamma(rgb * lumScale / lumRgb);
  }
  static glm::vec3 toReinhard2(const glm::vec3& rgb, float whiteSq) {
    float lumRgb = lumf(rgb);
    float lumScale = (lumRgb * (1.0f + lumRgb / whiteSq)) / (1.0f + lumRgb);
    return toGamma(rgb * lumScale / lumRgb);
  }
  static glm::vec3 toFilmic(const glm::vec3& rgb) {
    glm::vec3 res = glm::max(glm::vec3(0.0f), rgb - 0.004f);
    res = (res * (6.2f * res + 0.5f)) / (res * (6.2f * res + 1.7f) + 0.06f);
    return res;
  }
  static glm::vec3 toUncharted2b(const glm::vec3& rgb) {
    float exposureBias = 2.0f;
    glm::vec3 curr = exposureBias * toUncharted2(rgb);

    float w = 11.2f;
    glm::vec3 whiteScale = 1.0f / toUncharted2(w);

    const glm::vec3 rgb2 = curr * whiteScale;
    return toGamma(rgb2);
  }

private:
  static glm::vec3 toUncharted2(const glm::vec3& v) {
    float a = 0.22f;
    float b = 0.30f;
    float c = 0.10f;
    float d = 0.20f;
    float e = 0.01f;
    float f = 0.30f;
    //float w = 11.2f;
    return ((v * (a * v + c * b) + d * e) / (v * (a * v + b) + d * f)) - e / f;
  }

  static glm::vec3 toUncharted2(float x) {
    glm::vec3 v = glm::vec3(x);
    return toUncharted2(v);
  }

  static float lumf(glm::vec3 rgb) {
    return glm::dot(glm::vec3(0.2126729, 0.7151522, 0.0721750), rgb);
  }
};
}  // namespace ws
