#include "EmbreeUtils.hpp"

#include <stb_image.h>

#include <numbers>
#include <random>

namespace eu {

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

bool ERay::isOccluded() {
  rtcOccluded1(scene_, &rh_.ray);
  return rh_.ray.tfar < 0.f;
}



RTCGeometry makeTriangularGeometry(RTCDevice dev, const std::vector<glm::vec3>& verts, const std::vector<glm::vec3>& norms, const std::vector<glm::vec2>& texCoords, const std::vector<uint32_t>& ixs) {
  RTCGeometry geom = rtcNewGeometry(dev, RTC_GEOMETRY_TYPE_TRIANGLE);
  float* vertices = static_cast<float*>(rtcSetNewGeometryBuffer(geom,
                                                                RTC_BUFFER_TYPE_VERTEX,
                                                                0,
                                                                RTC_FORMAT_FLOAT3,
                                                                3 * sizeof(float),
                                                                verts.size()));
  unsigned* indices = static_cast<unsigned*>(rtcSetNewGeometryBuffer(geom,
                                                                     RTC_BUFFER_TYPE_INDEX,
                                                                     0,
                                                                     RTC_FORMAT_UINT3,
                                                                     3 * sizeof(unsigned),
                                                                     ixs.size()));
  std::memcpy(vertices, verts.data(), verts.size() * sizeof(glm::vec3));
  std::memcpy(indices, ixs.data(), ixs.size() * sizeof(uint32_t));

  rtcSetGeometryVertexAttributeCount(geom, 2);
  rtcSetSharedGeometryBuffer(geom, RTC_BUFFER_TYPE_VERTEX_ATTRIBUTE, 0, RTC_FORMAT_FLOAT3, norms.data(), 0, sizeof(glm::vec3), norms.size());
  rtcSetSharedGeometryBuffer(geom, RTC_BUFFER_TYPE_VERTEX_ATTRIBUTE, 1, RTC_FORMAT_FLOAT2, texCoords.data(), 0, sizeof(glm::vec2), texCoords.size());

  // only for instanced geometry
  // rtcSetGeometryTransform(geom, 0, RTC_FORMAT_FLOAT4X4_COLUMN_MAJOR, glm::value_ptr(xform));

  rtcCommitGeometry(geom);
  return geom;
}

// world vector to longitude latitude coordinates. singularity at (x=0 && z=0) requires disabling of mips or computing mip level manually
glm::vec2 dirToLonLat(glm::vec3 dir) {
  glm::vec2 vo{};
  vo.x = atan2(dir.x, dir.z) / std::numbers::pi_v<float>;
  vo.y = -dir.y;
  vo = vo * 0.5f + 0.5f;
  return vo;
}

float rndUni01() {
  static std::random_device rndDev;
  static std::mt19937 rndEngine(rndDev());
  static std::uniform_real_distribution<float> uni01(0.f, 1.0f);
  return uni01(rndEngine);
}

glm::vec3 sampleLambertian(const glm::vec3& norm, const glm::vec3& in, float spread) {
  float theta = 2.f * std::numbers::pi_v<float> * eu::rndUni01();
  float phi = std::acos(1.f - 2.f * eu::rndUni01());
  const glm::vec3 rndDir{
      sin(phi) * cos(theta),
      sin(phi) * sin(theta),
      cos(phi)};

  const glm::vec3 refl = glm::reflect(in, norm);
  return glm::normalize(norm + glm::mix(refl, rndDir, spread));
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
  int j = static_cast<int>(std::floor(x));
  int i = static_cast<int>(std::floor(y));
  glm::vec3 result = pixels[i * width + j];
  return result;
}
}  // namespace ws