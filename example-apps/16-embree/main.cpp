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
                                                                verts.size()));
  unsigned* indices = static_cast<unsigned*>(rtcSetNewGeometryBuffer(geom,
                                                                     RTC_BUFFER_TYPE_INDEX,
                                                                     0,
                                                                     RTC_FORMAT_UINT3,
                                                                     3 * sizeof(unsigned),
                                                                     ixs.size()));

  std::memcpy(vertices, verts.data(), verts.size() * sizeof(glm::vec3));
  std::memcpy(indices, ixs.data(), ixs.size() * sizeof(uint32_t));

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
  ws::Workshop workshop{800, 600, "Boilerplate app"};
  ws::AssetManager assetManager;
  assetManager.meshes.emplace("monkey", ws::loadOBJ(ws::ASSETS_FOLDER / "models/suzanne.obj"));
  assetManager.meshes.emplace("cube", ws::loadOBJ(ws::ASSETS_FOLDER / "models/cube.obj"));
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
  ws::Scene scene{
    .renderables{monkey},
  };
  ws::Framebuffer offscreenFbo = ws::Framebuffer::makeDefaultColorOnly(1, 1);

  ws::AutoOrbitingCameraController orbitingCamController{scene.camera};
  //orbitingCamController.radius = 10.f;
  //orbitingCamController.theta = 0.3f;

  /* for best performance set FTZ and DAZ flags in MXCSR control and status register */
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);

  RTCDevice device = rtcNewDevice("verbose=1");
  RTCScene eScene = rtcNewScene(device);

  std::vector<glm::vec3> verts; 
  for (const auto& v : assetManager.meshes.at("monkey").meshData.vertices)
    verts.push_back(v.position);  

  RTCGeometry geom = makeTriangularGeometry(device, verts, assetManager.meshes.at("monkey").meshData.indices);
  rtcAttachGeometry(eScene, geom);
  rtcReleaseGeometry(geom);
  rtcCommitScene(eScene);

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

    if (isRayTraced) {
      const int32_t width = winSize.x;
      const int32_t height = winSize.y;
      ws::Camera& cam = scene.camera;
      const glm::u8vec4 hitColor{255, 0, 0, 255};
      const glm::u8vec4 missColor{0, 0, 255, 255};
      std::vector<glm::u8vec4> pixels(width * height);
      for (int32_t i = 0; i < width; ++i) {
        for (int32_t j = 0; j < height; ++j) {
          float x = (i + 0.5f) / width - 0.5f;
          float y = (j + 0.5f) / height - 0.5f;

          const glm::vec3 forward = cam.getForward() * 0.5f / glm::tan(glm::radians(cam.fov) * 0.5f);
          const glm::vec3 right = cam.getRight() * cam.aspectRatio * x;
          const glm::vec3 up = cam.getUp() * y;
          const glm::vec3 d = glm::normalize(forward + right + up);
          const glm::vec3 o = cam.position;
          RTCRayHit rayHit = castRay(eScene, o, d);

          pixels[j * width + i] = (rayHit.hit.geomID != RTC_INVALID_GEOMETRY_ID) ? hitColor : missColor;
        }
      }
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