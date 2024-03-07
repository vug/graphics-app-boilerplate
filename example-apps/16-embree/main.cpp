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
    .tnear = 0.001,
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

RTCGeometry makeTriangularGeometry(RTCDevice dev, const std::vector<glm::vec3>& verts, const std::vector<glm::vec3>& norms, const std::vector<uint32_t>& ixs) {
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

  rtcSetGeometryVertexAttributeCount(geom, 1);
  rtcSetSharedGeometryBuffer(geom, RTC_BUFFER_TYPE_VERTEX_ATTRIBUTE, 0, RTC_FORMAT_FLOAT3, norms.data(), 0, sizeof(glm::vec3), norms.size());

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

glm::vec3 sampleLambertian(const glm::vec3& norm, float spread = 1.f) {
  float theta = 2.f * std::numbers::pi_v<float> * uniDist(rndEngine);
  float phi = std::acos(1.f - 2.f * uniDist(rndEngine));
  const glm::vec3 rndDir{
      sin(phi) * cos(theta),
      sin(phi) * sin(theta),
      cos(phi)};
  return glm::normalize(norm + glm::mix(norm, rndDir, spread));
}

float hable(float c) {
  float A = 0.15;
  float B = 0.50;
  float C = 0.10;
  float D = 0.20;
  float E = 0.02;
  float F = 0.30;

  return ((c * (A * c + C * B) + D * E) / (c * (A * c + B) + D * F)) - E / F;
}

glm::vec3 tonemap(glm::vec3 color) {
  // Calculate the desaturation coefficient based on the brightness
  float sig = std::max(color.r, std::max(color.g, color.b));
  float luma = glm::dot(color, glm::vec3(0.2126, 0.7152, 0.0722));
  float coeff = std::max(sig - 0.18, 1e-6) / std::max(sig, 1e-6f);
  coeff = std::powf(coeff, 20.0f);

  // Update the original color and signal
  color = glm::mix(color, glm::vec3(luma), coeff);
  sig = glm::mix(sig, luma, coeff);

  // Perform tone-mapping
  return color * hable(sig) / sig;
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
    {"Monkey", {glm::vec3{-2, 0, 0}, glm::vec3{1, 0, 0}, glm::radians(0.f), glm::vec3{2, 2, 2}}},
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
      {"Wall", {glm::vec3{0, -2, 0}, glm::vec3{1, 0, 0}, glm::radians(0.f), glm::vec3{20, 0.25, 20}}},
      assetManager.meshes.at("box"),
      assetManager.materials.at("solid_red"),
  };
  ws::Scene scene{
    .renderables{monkey, sphere, wall1},
  };
  std::vector<float> objEmissiveness = {0.0f, 0.0f, 0.f};
  std::vector<float> objRoughnesses = {0.5f, 0.0f, 1.f};
  std::vector<glm::vec3> objColors = {{1.f, 0.8f, 0.6f},
                                      {0.8f, 0.6f, 1.f},
                                      {0.75f, 0.75f, 0.75f}};
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

  // Store world normals in a vector because we want them to exist until we exit the app. rtcSetSharedGeometryBuffer does not copy normals into a new memory location.
  std::vector<std::vector<glm::vec3>> worldNormalsVec;
  for (const auto& r : scene.renderables) {
    const auto& xform = r.get().getGlobalTransformMatrix();
    const glm::mat3 invTranspXForm = glm::mat3(glm::transpose(glm::inverse(xform)));

    std::vector<glm::vec3> worldPositions;
    std::vector<glm::vec3>& worldNormals = worldNormalsVec.emplace_back();
    for (const auto& v : r.get().mesh.meshData.vertices) {
      const glm::vec3 worldPosition = glm::vec3(xform * glm::vec4(v.position, 1));
      worldPositions.push_back(worldPosition);
      const glm::vec3 worldNormal = invTranspXForm * v.normal;
      worldNormals.push_back(worldNormal);
    }

    RTCGeometry geom = makeTriangularGeometry(device, worldPositions, worldNormals, r.get().mesh.meshData.indices);
    rtcAttachGeometry(eScene, geom);
    rtcReleaseGeometry(geom);
  }
  rtcCommitScene(eScene);

  //glEnable(GL_DEPTH_TEST);
  glEnable(GL_CULL_FACE);
  scene.ubo.compareSizeWithUniformBlock(assetManager.shaders.at("solid_color").getId(), "SceneUniforms");

  float numFrames = 1.f;
  int numAccumulatedSamplesPerPixel = 0;
  while (!workshop.shouldStop()) {
    workshop.beginFrame();
    const glm::uvec2 winSize = workshop.getWindowSize();
    offscreenFbo.resizeIfNeeded(winSize.x, winSize.y);

    bool hasChanged = false;
    ImGui::Begin("Embree Path Tracer Study");
    static glm::vec3 skyColorNorth{0.5, 0.7, 1.0};
    hasChanged |= ImGui::ColorEdit3("Sky Color North", glm::value_ptr(skyColorNorth));
    static glm::vec3 skyColorSouth{1};
    hasChanged |= ImGui::ColorEdit3("Sky Color South", glm::value_ptr(skyColorSouth));
    static float skyEmissive{1.0f};
    hasChanged |= ImGui::SliderFloat("Sky Emissive", &skyEmissive, 0.f, 4.f);
    static int32_t numMaxBounces = 4;
    hasChanged |= ImGui::SliderInt("Num Max Hits", &numMaxBounces, 0, 8);
    static int32_t numSamplesPerPixel = 1;
    hasChanged |= ImGui::SliderInt("Num Samples Per Pixel", &numSamplesPerPixel, 1, 8);
    //hasChanged |= ImGui::DragFloat3("Obj1 Position", glm::value_ptr(scene.renderables[0].get().transform.position));
    hasChanged |= ImGui::ColorEdit3("Obj1 Color", glm::value_ptr(objColors[0]));
    hasChanged |= ImGui::SliderFloat("Obj1 Emissive", &objEmissiveness[0], 0.f, 4.f);
    hasChanged |= ImGui::SliderFloat("Obj1 Rougness", &objRoughnesses[0], 0.f, 1.f);
    hasChanged |= ImGui::ColorEdit3("Obj2 Color", glm::value_ptr(objColors[1]));
    hasChanged |= ImGui::SliderFloat("Obj2 Emissive", &objEmissiveness[1], 0.f, 4.f);
    hasChanged |= ImGui::SliderFloat("Obj2 Rougness", &objRoughnesses[1], 0.f, 1.f);
    hasChanged |= ImGui::ColorEdit3("Obj3 Color", glm::value_ptr(objColors[2]));
    hasChanged |= ImGui::SliderFloat("Obj3 Emissive", &objEmissiveness[2], 0.f, 4.f);
    hasChanged |= ImGui::SliderFloat("Obj3 Rougness", &objRoughnesses[2], 0.f, 1.f);
    static float roughness = 1.f;
    hasChanged |= ImGui::SliderFloat("Roughness", &roughness, 0.f, 1.f);

    ImGui::Separator();
    static bool isRayTraced = true;
    ImGui::Checkbox("raytraced?", &isRayTraced);
    ImGui::Text("Samples accumulated %d", numAccumulatedSamplesPerPixel);
    ImGui::End();

    bool hasCameraMoved = orbitingCamController.update(workshop.getFrameDurationSec());
    scene.camera.aspectRatio = static_cast<float>(winSize.x) / winSize.y;
    scene.uploadUniforms();

    const glm::vec3 lightPos{10, 10, 10};

    const int32_t width = winSize.x;
    const int32_t height = winSize.y;
    static std::vector<glm::u8vec4> pixels(width * height);
    static std::vector<glm::vec3> pixelColors(width * height);
    const bool shouldResizePixelBuffers = pixels.size() != width * height;
    if (shouldResizePixelBuffers) {
      pixels.resize(width * height);
      pixelColors.resize(width * height);
    }
    if (hasChanged || hasCameraMoved || shouldResizePixelBuffers) {
      std::ranges::fill(pixels, glm::u8vec4(0, 0, 0, 1));
      std::ranges::fill(pixelColors, glm::vec3(0));
      numAccumulatedSamplesPerPixel = 0;
      numFrames = 1.f;
    }
    if (isRayTraced) {
      ws::Camera& cam = scene.camera;
      for (int32_t i = 0; i < width; ++i) {
        for (int32_t j = 0; j < height; ++j) {
          glm::vec3 color{};
          for (int32_t s = 0; s < numSamplesPerPixel; ++s) {
            glm::vec3 sampleColor{};
            glm::vec3 attenuation{1, 1, 1};
            float x = (i + uniDist(rndEngine)) / width - 0.5f;
            float y = (j + uniDist(rndEngine)) / height - 0.5f;

            const glm::vec3 forward = cam.getForward() * 0.5f / glm::tan(glm::radians(cam.fov) * 0.5f);
            const glm::vec3 right = cam.getRight() * cam.aspectRatio * x;
            const glm::vec3 up = cam.getUp() * y;
          
            glm::vec3 d = glm::normalize(forward + right + up);
            glm::vec3 o = cam.position;
            for (int32_t k = 0; k < numMaxBounces; ++k) {
              RTCRayHit rayHit = makeRayHit(o, d);
              rtcIntersect1(eScene, &rayHit);
              const uint32_t geoId = rayHit.hit.geomID;
              if (geoId == RTC_INVALID_GEOMETRY_ID) {
                const float m = 0.5f * (d.y + 1.0f);
                const glm::vec3 skyColor = glm::mix(skyColorSouth, skyColorNorth, m);
                color += attenuation * skyColor * skyEmissive;
                break;
              }

              auto geo = rtcGetGeometry(eScene, geoId);
              glm::vec3 normal;
              rtcInterpolate0(geo, rayHit.hit.primID, rayHit.hit.u, rayHit.hit.v, RTC_BUFFER_TYPE_VERTEX_ATTRIBUTE, 0, glm::value_ptr(normal), 3);
              normal = glm::normalize(normal);

              glm::vec3 pos;
              pos = {o + d * rayHit.ray.tfar};

              // sampleColor = normal * 0.5f + 0.5f;
              // sampleColor = pos;
              // sampleColor = glm::max(glm::dot(glm::normalize(lightPos - pos), normal), 0.f) * glm::vec3(1);

              sampleColor += attenuation * objEmissiveness[geoId] * objColors[geoId];
              attenuation *= objColors[geoId];

              // rebounce
              o = pos;
              //d = sampleHemisphere(normal);
              d = sampleLambertian(normal, objRoughnesses[geoId]);
              //d = glm::reflect(d, normal);
            }
            color += sampleColor;
          }
          color /= numSamplesPerPixel;

          const size_t ix = j * width + i;
          pixelColors[ix] += color;
          //pixelColors[ix] = tonemap(color);
          pixels[ix] = glm::u8vec4{glm::clamp(pixelColors[ix] / numFrames, 0.f, 1.f) * 255.f, 255};
        }
      }
      ++numFrames;
      numAccumulatedSamplesPerPixel += numSamplesPerPixel;
      offscreenFbo.getFirstColorAttachment().uploadPixels(pixels.data());

      glViewport(0, 0, winSize.x, winSize.y);
      offscreenFbo.getFirstColorAttachment().bindToUnit(0);
      assetManager.shaders.at("copy").bind();
      assetManager.drawWithEmptyVao(6);
      ws::Shader::unbind();
      ws::Texture::unbindFromUnit(0);
    } else {
      glViewport(0, 0, winSize.x, winSize.y);
      ws::Framebuffer::clear(0, glm::vec4(skyColorNorth, 1.f));
      scene.draw();
    }

    workshop.drawUI();

    workshop.endFrame();
  }

  rtcReleaseScene(eScene);
  rtcReleaseDevice(device);
  return 0;
}