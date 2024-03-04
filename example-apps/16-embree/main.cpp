#include <Workshop/Camera.hpp>

#include <embree4/rtcore.h>
#include <glm/glm.hpp>
#include <stb/stb_image_write.h>

#include "pmmintrin.h"
#include "xmmintrin.h"

#include <print>
#include <vector>

RTCRayHit castRay(RTCScene scene, const glm::vec3& o, const glm::vec3& d) {
  struct RTCRayHit rayhit;
  rayhit.ray.org_x = o.x;
  rayhit.ray.org_y = o.y;
  rayhit.ray.org_z = o.z;
  rayhit.ray.dir_x = d.x;
  rayhit.ray.dir_y = d.y;
  rayhit.ray.dir_z = d.z;
  rayhit.ray.tnear = 0;
  rayhit.ray.tfar = std::numeric_limits<float>::infinity();
  rayhit.ray.mask = static_cast<unsigned>(-1);
  rayhit.ray.flags = 0;
  rayhit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
  rayhit.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;

  rtcIntersect1(scene, &rayhit);

  return rayhit;
}

RTCGeometry makeTriangularGeometry(RTCDevice dev, const std::vector<glm::vec3>& verts, const std::vector<uint32_t>& ixs) {
  RTCGeometry geom = rtcNewGeometry(dev, RTC_GEOMETRY_TYPE_TRIANGLE);
  float* vertices = static_cast<float*>(rtcSetNewGeometryBuffer(geom,
                                                                RTC_BUFFER_TYPE_VERTEX,
                                                                0,
                                                                RTC_FORMAT_FLOAT3,
                                                                3 * sizeof(float),
                                                                3));
  unsigned* indices = static_cast<unsigned*>(rtcSetNewGeometryBuffer(geom,
                                                                     RTC_BUFFER_TYPE_INDEX,
                                                                     0,
                                                                     RTC_FORMAT_UINT3,
                                                                     3 * sizeof(unsigned),
                                                                     1));

  for (int i = 0; i < verts.size(); ++i)
    for (int j = 0; j < 3; ++j)
      vertices[3 * i + j] = verts[i][j];
  for (unsigned ix : ixs)
    indices[ix] = ix;
  rtcCommitGeometry(geom);
  return geom;
}

void handleHit(const RTCRayHit rayhit) {
  std::print("({}, {}, {}): ", rayhit.ray.org_x, rayhit.ray.org_y, rayhit.ray.org_z);
  if (rayhit.hit.geomID != RTC_INVALID_GEOMETRY_ID)
    std::println("Found intersection on geometry {}, primitive {} at tfar={}, normal=({}, {}, {}), uv=({}, {})",
                 rayhit.hit.geomID,
                 rayhit.hit.primID,
                 rayhit.ray.tfar,
                 rayhit.hit.Ng_x, rayhit.hit.Ng_y, rayhit.hit.Ng_z,
                 rayhit.hit.u, rayhit.hit.v);
  else
    std::println("Did not find any intersection.");
}

int main() {
  /* for best performance set FTZ and DAZ flags in MXCSR control and status register */
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);

  RTCDevice device = rtcNewDevice("verbose=1");
  RTCScene scene = rtcNewScene(device);

  const std::vector<glm::vec3> verts{
      {0, 0, 0},
      {1, 0, 0},
      {0, 1, 0},
  };
  const std::vector<uint32_t> ixs{0, 1, 2};

  RTCGeometry geom = makeTriangularGeometry(device, verts, ixs);
  rtcAttachGeometry(scene, geom);
  rtcReleaseGeometry(geom);
  rtcCommitScene(scene);

  const int32_t width = 800;
  const int32_t height = 600;
  ws::Camera cam;
  cam.position = {0, 0, 5};
  cam.target = {0, 0, 0};
  cam.aspectRatio = static_cast<float>(width) / height;
  const glm::u8vec3 hitColor{255, 0, 0};
  const glm::u8vec3 missColor{0, 0, 255};
  std::vector<glm::u8vec3> pixels(width * height);
  for (int32_t i = 0; i < width; ++i) {
    for (int32_t j = 0; j < height; ++j) {
      float x = (i + 0.5f) / width - 0.5f;
      float y = (j + 0.5f) / height - 0.5f;

      const glm::vec3 forward = cam.getForward() * 0.5f / glm::tan(glm::radians(cam.fov) * 0.5f);
      const glm::vec3 right = cam.getRight() * cam.aspectRatio * x;
      const glm::vec3 up = cam.getUp() * y;
      const glm::vec3 d = glm::normalize(forward + right + up);
      const glm::vec3 o = cam.position;
      RTCRayHit rayHit = castRay(scene, o, d);

      pixels[(height - j - 1) * width + i] = (rayHit.hit.geomID != RTC_INVALID_GEOMETRY_ID) ? hitColor : missColor;
    }
  }
  stbi_write_png("raytraced.png", width, height, 3, pixels.data(), sizeof(uint8_t) * width * 3);

  rtcReleaseScene(scene);
  rtcReleaseDevice(device);
  return 0;
}