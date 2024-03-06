#include <Workshop/AssetManager.hpp>
#include <Workshop/Assets.hpp>
#include <Workshop/Camera.hpp>
#include <Workshop/Framebuffer.hpp>
#include <Workshop/Scene.hpp>
#include <Workshop/Workshop.hpp>

#include <embree4/rtcore.h>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <imgui.h>

#include "pmmintrin.h"
#include "xmmintrin.h"

#include <print>
#include <random>
#include <vector>

RTCRay makeRay(const glm::vec3& o, const glm::vec3& d) {
  RTCRay r{
    .org_x = o.x,
    .org_y = o.y,
    .org_z = o.z,
    .tnear = 0,
    .dir_x = d.x,
    .dir_y = d.y,
    .dir_z = d.z,
    .tfar = std::numeric_limits<float>::infinity(),
    .mask = static_cast<unsigned>(-1),
    .flags = 0,
  };
  return r;
}

RTCHit makeHit() {
  RTCHit h;
  h.geomID = RTC_INVALID_GEOMETRY_ID;
  h.instID[0] = RTC_INVALID_GEOMETRY_ID;
  return h;
}

RTCRayHit makeRayHit(const glm::vec3& o, const glm::vec3& d) {
  RTCRayHit rh;
  rh.ray = makeRay(o, d);
  rh.hit = makeHit();
  return rh;
}

RTCGeometry makeTriangularGeometry(RTCDevice dev, const std::vector<glm::vec3>& verts, const std::vector<glm::vec3>& norms, const std::vector<uint32_t>& ixs, const glm::mat4& xform) {
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
  const glm::mat3 invTranspXForm = glm::mat3(glm::transpose(glm::inverse(xform)));

  std::vector<glm::vec3> worldPositions;
  std::vector<glm::vec3> worldNormals;
  for (int32_t i = 0; i < verts.size(); ++i) {
    // v.worldPosition = vec3(worldFromObject * vec4(objectPosition, 1));
    const glm::vec3 worldPosition = glm::vec3(xform * glm::vec4(verts[i], 1));
    worldPositions.push_back(worldPosition);

    // v.worldNormal = mat3(transpose(inverse(worldFromObject))) * objectNormal;
    const glm::vec3 worldNormal = invTranspXForm * norms[i];
    worldNormals.push_back(worldNormal);
  }
  std::memcpy(vertices, worldPositions.data(), worldPositions.size() * sizeof(glm::vec3));
  std::memcpy(indices, ixs.data(), ixs.size() * sizeof(uint32_t));

  rtcSetGeometryVertexAttributeCount(geom, 1);
  rtcSetSharedGeometryBuffer(geom, RTC_BUFFER_TYPE_VERTEX_ATTRIBUTE, 0, RTC_FORMAT_FLOAT3, worldNormals.data(), 0, sizeof(glm::vec3), worldNormals.size());

  // only for instanced geometry
  //rtcSetGeometryTransform(geom, 0, RTC_FORMAT_FLOAT4X4_COLUMN_MAJOR, glm::value_ptr(xform));

  rtcCommitGeometry(geom);
  return geom;
}

std::random_device rndDev;
std::mt19937 rndEngine(rndDev());
std::uniform_real_distribution<float> uniDistCircle(0.f, std::numbers::pi_v<float>);
std::uniform_real_distribution<float> uniDist(0.f, 1.0f);

glm::vec3 sampleHemisphere(const glm::vec3& norm) {
  // surface coordinate system
  const glm::vec3& Z = norm;
  glm::vec3 axis;
  if (Z.x < Z.y && Z.x < Z.z)
    axis = {1.0f, 0.0f, 0.0f};
  else if (Z.y < Z.z)
    axis = {0.0f, 1.0f, 0.0f};
  else
    axis = {0.0f, 0.0f, 1.0f};
  const glm::vec3 X = glm::normalize(glm::cross(Z, axis));
  const glm::vec3 Y = glm::cross(Z, X);

  // sample unit disk
  const float phi = uniDistCircle(rndEngine);
  const float r = uniDist(rndEngine);
  const float r2 = r * r;
  const float x = r2 * std::cos(phi);
  const float y = r2 * std::sin(phi);

  // project to hemisphere
  const float z = std::sqrtf(1 - x * x - y * y);
  const glm::vec3 refDir = X * x + Y * y + Z * z;

  return refDir;
}

int main() {
  ws::Workshop workshop{800, 600, "Embree Path Tracer Study"};
  ws::AssetManager assetManager;
  assetManager.meshes.emplace("monkey", ws::loadOBJ(ws::ASSETS_FOLDER / "models/suzanne_smooth.obj"));
  assetManager.meshes.emplace("sphere", ws::loadOBJ(ws::ASSETS_FOLDER / "models/sphere_ico_smooth.obj"));
  assetManager.meshes.emplace("box", ws::loadOBJ(ws::ASSETS_FOLDER / "models/cube.obj"));
  assetManager.shaders.emplace("solid_color", ws::Shader{ws::ASSETS_FOLDER / "shaders/solid_color.vert", ws::ASSETS_FOLDER / "shaders/solid_color.frag"});
  assetManager.shaders.emplace("copy", ws::Shader{ws::ASSETS_FOLDER / "shaders/fullscreen_quad_without_vbo.vert", ws::ASSETS_FOLDER / "shaders/fullscreen_quad_texture_sampler.frag"});
  assetManager.materials.emplace("solid_red", ws::Material{
                                                  .shader = assetManager.shaders.at("solid_color"),
                                                  .parameters = {
                                                      {"u_Color", glm::vec4{1, 0, 0, 1}}}});
  assert(assetManager.doAllMaterialsHaveMatchingParametersAndUniforms());
  ws::RenderableObject monkey = {
    //{"Monkey", {glm::vec3{0, 0, 0}, glm::vec3{1, 0, 0}, glm::radians(-30.f), glm::vec3{1.5f, 1.5f, 1.5f}}},
    {"Monkey", {glm::vec3{0, 0, 0}, glm::vec3{1, 0, 0}, glm::radians(0.f), glm::vec3{1, 1, 1}}},
    assetManager.meshes.at("monkey"),
    assetManager.materials.at("solid_red"),
  };
  ws::RenderableObject sphere = {
      //{"Monkey", {glm::vec3{0, 0, 0}, glm::vec3{1, 0, 0}, glm::radians(-30.f), glm::vec3{1.5f, 1.5f, 1.5f}}},
      {"Sphere", {glm::vec3{3, 0, 0}, glm::vec3{1, 0, 0}, glm::radians(0.f), glm::vec3{2, 2, 2}}},
      assetManager.meshes.at("sphere"),
      assetManager.materials.at("solid_red"),
  };
  ws::RenderableObject wall1 = {
      //{"Monkey", {glm::vec3{0, 0, 0}, glm::vec3{1, 0, 0}, glm::radians(-30.f), glm::vec3{1.5f, 1.5f, 1.5f}}},
      {"Wall", {glm::vec3{-3, 0, 0}, glm::vec3{1, 0, 0}, glm::radians(0.f), glm::vec3{0.5, 10, 10}}},
      assetManager.meshes.at("box"),
      assetManager.materials.at("solid_red"),
  };
  ws::Scene scene{
    .renderables{monkey, sphere, wall1},
  };
  std::vector<float> objEmissiveness = {0.f, 5.0f, 0.f};
  std::vector<glm::vec3> objColors = {{1.f, 0.8f, 0.6f},
                                      {1, 1, 1},
                                      {1, 1, 1}};
  ws::Framebuffer offscreenFbo = ws::Framebuffer::makeDefaultColorOnly(1, 1);

  ws::AutoOrbitingCameraController orbitingCamController{scene.camera};
  orbitingCamController.speed = 0;
  orbitingCamController.phi0 = 1;
  orbitingCamController.radius = 10;

  /* for best performance set FTZ and DAZ flags in MXCSR control and status register */
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);

  RTCDevice device = rtcNewDevice("verbose=1");
  RTCScene eScene = rtcNewScene(device);

  for (const auto& r : scene.renderables) {
    std::vector<glm::vec3> verts; 
    for (const auto& v : r.get().mesh.meshData.vertices)
      verts.push_back(v.position);  
    std::vector<glm::vec3> norms; 
    for (const auto& v : r.get().mesh.meshData.vertices)
      norms.push_back(v.normal);  

    RTCGeometry geom = makeTriangularGeometry(device, verts, norms, r.get().mesh.meshData.indices, r.get().getGlobalTransformMatrix());
    rtcAttachGeometry(eScene, geom);
    rtcReleaseGeometry(geom);
    rtcCommitScene(eScene);
  }

  //glEnable(GL_DEPTH_TEST);
  glEnable(GL_CULL_FACE);
  scene.ubo.compareSizeWithUniformBlock(assetManager.shaders.at("solid_color").getId(), "SceneUniforms");

  while (!workshop.shouldStop()) {
    workshop.beginFrame();
    const glm::uvec2 winSize = workshop.getWindowSize();
    offscreenFbo.resizeIfNeeded(winSize.x, winSize.y);

    ImGui::Begin("Boilerplate");
    static glm::vec4 bgColor{42 / 256.f, 96 / 256.f, 87 / 256.f, 1.f};
    ImGui::ColorEdit4("BG Color", glm::value_ptr(bgColor));
    ImGui::Separator();
    static bool isRayTraced = true;
    ImGui::Checkbox("raytraced?", &isRayTraced);
    ImGui::End();

    orbitingCamController.update(workshop.getFrameDurationSec());
    scene.camera.aspectRatio = static_cast<float>(winSize.x) / winSize.y;
    scene.uploadUniforms();

    const glm::vec3 lightPos{10, 10, 10};

    const int32_t width = winSize.x;
    const int32_t height = winSize.y;
    static std::vector<glm::u8vec4> pixels(width * height);
    static std::vector<glm::vec3> pixelColors(width * height);
    float numFrames = 1.f;
    if (isRayTraced) {
      const int32_t numMaxBounces = 2;
      const int32_t numSamplesPerPixel = 3;

      ws::Camera& cam = scene.camera;
      const glm::u8vec4 missColor{0, 0, 255, 255};
      for (int32_t i = 0; i < width; ++i) {
        for (int32_t j = 0; j < height; ++j) {
          float x = (i + uniDist(rndEngine)) / width - 0.5f;
          float y = (j + uniDist(rndEngine)) / height - 0.5f;

          const glm::vec3 forward = cam.getForward() * 0.5f / glm::tan(glm::radians(cam.fov) * 0.5f);
          const glm::vec3 right = cam.getRight() * cam.aspectRatio * x;
          const glm::vec3 up = cam.getUp() * y;
          glm::vec3 color{};
          
          for (int32_t s = 0; s < numSamplesPerPixel; ++s) {
            glm::vec3 d = glm::normalize(forward + right + up);
            glm::vec3 o = cam.position;
            glm::vec3 attenuation{1, 1, 1};
            for (int32_t k = 0; k < numMaxBounces; ++k) {
              RTCRayHit rayHit = makeRayHit(o, d);
              rtcIntersect1(eScene, &rayHit);
              const uint32_t geoId = rayHit.hit.geomID;
              if (geoId == RTC_INVALID_GEOMETRY_ID) {
                color += glm::vec3(0.0f);
                attenuation = {0, 0, 0};
                break;
              }

              auto geo = rtcGetGeometry(eScene, geoId);
              glm::vec3 normal;
              rtcInterpolate0(geo, rayHit.hit.primID, rayHit.hit.u, rayHit.hit.v, RTC_BUFFER_TYPE_VERTEX_ATTRIBUTE, 0, glm::value_ptr(normal), 3);
              normal = glm::normalize(normal);

              glm::vec3 pos;
              pos = {o + d * rayHit.ray.tfar};

              const float objEmis = objEmissiveness[geoId];
              const glm::vec3 objCol = objColors[geoId];
              color += glm::vec3(objEmis) * attenuation;
              attenuation *= objCol;
              o = pos;
              d = sampleHemisphere(normal);
            }
          }
          color /= numSamplesPerPixel;

          const size_t ix = j * width + i;
          pixelColors[ix] = color;
          pixels[ix] = glm::u8vec4{pixelColors[ix] / numFrames * 255.f, 255};
        }
      }
      ++numFrames;
      offscreenFbo.getFirstColorAttachment().uploadPixels(pixels.data());
      //rtcSetGeometryTransform(geom, workshop.getFrameNo(), RTC_FORMAT_FLOAT4X4_COLUMN_MAJOR, glm::value_ptr(monkey.getGlobalTransformMatrix()));

      glViewport(0, 0, winSize.x, winSize.y);
      ws::Framebuffer::clear(0, bgColor);
      offscreenFbo.getFirstColorAttachment().bindToUnit(0);
      assetManager.shaders.at("copy").bind();
      assetManager.drawWithEmptyVao(6);
      ws::Shader::unbind();
      ws::Texture::unbindFromUnit(0);
    } else {
      glViewport(0, 0, winSize.x, winSize.y);
      ws::Framebuffer::clear(0, bgColor);
      scene.draw();
    }

    workshop.endFrame();
  }

  rtcReleaseScene(eScene);
  rtcReleaseDevice(device);
  return 0;
}