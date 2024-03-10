#include "EmbreeUtils.hpp"

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
#include <ranges>
#include <vector>

std::random_device rndDev;
std::mt19937 rndEngine(rndDev());
std::uniform_real_distribution<float> uniDistCircle(0.f, std::numbers::pi_v<float>);
std::uniform_real_distribution<float> uniDist(0.f, 1.0f);

glm::vec3 sampleLambertian(const glm::vec3& norm, const glm::vec3& in, float spread = 1.f) {
  float theta = 2.f * std::numbers::pi_v<float> * uniDist(rndEngine);
  float phi = std::acos(1.f - 2.f * uniDist(rndEngine));
  const glm::vec3 rndDir{
      sin(phi) * cos(theta),
      sin(phi) * sin(theta),
      cos(phi)};

  const glm::vec3 refl = glm::reflect(in, norm);
  return glm::normalize(norm + glm::mix(refl, rndDir, spread));
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
    {"Monkey", {glm::vec3{-1.5, 0, 0}, glm::vec3{1, 0, 0}, glm::radians(0.f), glm::vec3{2, 2, 2}}},
    assetManager.meshes.at("monkey"),
    assetManager.materials.at("solid_red"),
  };
  ws::RenderableObject sphere = {
      //{"Monkey", {glm::vec3{0, 0, 0}, glm::vec3{1, 0, 0}, glm::radians(-30.f), glm::vec3{1.5f, 1.5f, 1.5f}}},
      {"Sphere", {glm::vec3{2.5, 0, 0}, glm::vec3{1, 0, 0}, glm::radians(0.f), glm::vec3{2, 2, 2}}},
      assetManager.meshes.at("sphere"),
      assetManager.materials.at("solid_red"),
  };
  ws::RenderableObject ground = {
      //{"Monkey", {glm::vec3{0, 0, 0}, glm::vec3{1, 0, 0}, glm::radians(-30.f), glm::vec3{1.5f, 1.5f, 1.5f}}},
      {"Wall", {glm::vec3{0, -2, 0}, glm::vec3{1, 0, 0}, glm::radians(0.f), glm::vec3{20, 0.25, 20}}},
      assetManager.meshes.at("box"),
      assetManager.materials.at("solid_red"),
  };
  ws::Scene scene{
    .renderables{monkey, sphere, ground},
  };
  std::vector<float> objEmissiveness = {0.0f, 0.0f, 0.0f};
  std::vector<float> objRoughnesses = {1.0f, 1.0f, 1.0f};
  std::vector<glm::vec3> objColors = {{1.f, 0.8f, 0.6f},
                                      {0.8f, 0.6f, 1.f},
                                      {0.75f, 0.75f, 0.75f}};
  std::vector<eu::Image> objAlbedos;
  //objAlbedos.emplace_back(ws::ASSETS_FOLDER / "images/LearnOpenGL/container.jpg");
  objAlbedos.emplace_back(ws::ASSETS_FOLDER / "images/Wikipedia/UV_checker_Map_byValle_1024.jpg");
  objAlbedos.emplace_back(ws::ASSETS_FOLDER / "images/LearnOpenGL/metal.png");
  objAlbedos.emplace_back(ws::ASSETS_FOLDER / "images/LearnOpenGL/brickwall.jpg");
  // Skybox textures
  std::vector<eu::Image> skyboxes;
  skyboxes.emplace_back(ws::ASSETS_FOLDER / "images/hdr/the_lost_city.hdr");
  skyboxes.emplace_back(ws::ASSETS_FOLDER / "images/hdr/little_paris.hdr");
  skyboxes.emplace_back(ws::ASSETS_FOLDER / "images/hdr/night_snowy_christmas.hdr");
  std::vector<eu::Image> skyboxIrradiances;
  skyboxIrradiances.emplace_back(ws::ASSETS_FOLDER / "images/hdr/the_lost_city_irradiance.hdr");
  skyboxIrradiances.emplace_back(ws::ASSETS_FOLDER / "images/hdr/little_paris_irradiance.hdr");
  skyboxIrradiances.emplace_back(ws::ASSETS_FOLDER / "images/hdr/night_snowy_christmas_irradiance.hdr");
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
  std::vector<std::vector<glm::vec2>> texCoordss;
  for (const auto& r : scene.renderables) {
    const auto& xform = r.get().getGlobalTransformMatrix();
    const glm::mat3 invTranspXForm = glm::mat3(glm::transpose(glm::inverse(xform)));

    std::vector<glm::vec3> worldPositions;
    std::vector<glm::vec3>& worldNormals = worldNormalsVec.emplace_back();
    std::vector<glm::vec2>& texCoords = texCoordss.emplace_back();
    for (const auto& v : r.get().mesh.meshData.vertices) {
      const glm::vec3 worldPosition = glm::vec3(xform * glm::vec4(v.position, 1));
      worldPositions.push_back(worldPosition);
      const glm::vec3 worldNormal = invTranspXForm * v.normal;
      worldNormals.push_back(worldNormal);
      texCoords.push_back(v.texCoord);

    }

    RTCGeometry geom = eu::makeTriangularGeometry(device, worldPositions, worldNormals, texCoords, r.get().mesh.meshData.indices);
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
    //static glm::vec3 skyColorNorth{0.5, 0.7, 1.0};
    //hasChanged |= ImGui::ColorEdit3("Sky Color North", glm::value_ptr(skyColorNorth));
    //static glm::vec3 skyColorSouth{1};
    //hasChanged |= ImGui::ColorEdit3("Sky Color South", glm::value_ptr(skyColorSouth));
    const std::array<const char*, 3> skyNames = {"ruins", "paris", "night"};
    static int skyboxIx = 0;
    hasChanged |= ImGui::Combo("Skybox", &skyboxIx, skyNames.data(), static_cast<int>(skyNames.size()));
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

    ImGui::Separator();
    static bool isRayTraced = true;
    ImGui::Checkbox("raytraced?", &isRayTraced);
    const std::array<const char*, 5> vizOpts = {"Scene", "Pos", "Normal", "UV", "Phong"};
    static int vizOpt = 0;
    hasChanged |= ImGui::Combo("Shading Mode", &vizOpt, vizOpts.data(), static_cast<int>(vizOpts.size()));
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
      for (auto [i, j] : std::views::cartesian_product(std::views::iota(0, width), std::views::iota(0, height))) {
        glm::vec3 pixCol{};
        for (int32_t s = 0; s < numSamplesPerPixel; ++s) {
          glm::vec3 sampCol{0};
          glm::vec3 attenuation{1};

          float x = (i + uniDist(rndEngine)) / width - 0.5f;
          float y = (j + uniDist(rndEngine)) / height - 0.5f;
          glm::vec3 d = cam.getRayDirection(x, y);
          glm::vec3 o = cam.position;
          for (int32_t k = 0; k < numMaxBounces; ++k) {
            eu::ERay ray(eScene, o, d);
            const eu::ERayResult res = ray.intersect();
            if (res.hasMissed) {
              // procedural sky
              //const float m = 0.5f * (res.direction.y + 1.0f);
              //const glm::vec3 skyColor = glm::mix(skyColorSouth, skyColorNorth, m);
              //sampCol += attenuation * skyColor * skyEmissive;

              glm::vec2 lonLat = eu::dirToLonLat(res.direction);
              const eu::Image& img = (k == 0) ? skyboxes[skyboxIx] : skyboxIrradiances[skyboxIx];
              sampCol += attenuation * img.nearest(lonLat.x * img.getWidth(), lonLat.y * img.getHeight()) * skyEmissive;
              break;
            }

            const glm::vec3 normal = glm::normalize(res.interpolateVertexAttribute<glm::vec3>(0));
            const glm::vec2 texCoord = res.interpolateVertexAttribute<glm::vec2>(1);
            //const glm::vec3 objColor = objColors[res.geomId]; // solid color albedo
            //const glm::vec3 objColor = glm::vec3(texCoord, 0); // use UVs as albedo
            const eu::Image& img = objAlbedos[res.geomId];  // use textures for albedo
            glm::vec3 objColor = img.nearest(texCoord.x * img.getWidth(), texCoord.y * img.getHeight());

            switch (vizOpt) {
              case 0:
                sampCol += attenuation * objEmissiveness[res.geomId] * objColor;
                break;
              case 1:
                sampCol = res.position;
                break;
              case 2:
                sampCol = normal * 0.5f + 0.5f;
                break;
              case 3:
                sampCol = glm::vec3(texCoord, 0);
                break;
              case 4:
                sampCol = glm::max(glm::dot(glm::normalize(lightPos - res.position), normal), 0.f) * glm::vec3(1);
                break;
              default:
                std::unreachable();
            }
            if (vizOpt != 0) break; // don't bounce if debug viz
            attenuation *= objColor;

            // rebounce
            d = sampleLambertian(normal, res.direction, objRoughnesses[res.geomId]);
            //d =  glm::reflect(d, normal);
            o = res.position;
          }
          pixCol += sampCol;
        }
        pixCol /= numSamplesPerPixel;

        const size_t ix = j * width + i;
        pixelColors[ix] += pixCol;
        pixels[ix] = glm::u8vec4{glm::clamp(pixelColors[ix] / numFrames, 0.f, 1.f) * 255.f, 255};
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
      ws::Framebuffer::clear(0, glm::vec4(1.f, 1.f, 1.f, 1.f));
      scene.draw();
    }

    workshop.drawUI();

    workshop.endFrame();
  }

  rtcReleaseScene(eScene);
  rtcReleaseDevice(device);
  return 0;
}