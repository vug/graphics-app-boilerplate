#include <Workshop/Assets.hpp>
#include <Workshop/AssetManager.hpp>
#include <Workshop/Camera.hpp>
#include <Workshop/Framebuffer.hpp>
#include <Workshop/Model.hpp>
#include <Workshop/Scene.hpp>
#include <Workshop/Shader.hpp>
#include <Workshop/UI.hpp>
#include <Workshop/Workshop.hpp>

#include <glad/gl.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <glm/gtc/type_ptr.hpp>
#include <glm/vec3.hpp>

#include <print>
#include <ranges>
#include <string>
#include <vector>

const std::filesystem::path SRC{SOURCE_DIR};

int main() {
  std::println("Hi!");
  ws::Workshop workshop{1920, 1080, "Bloom"};

  ws::AssetManager assetManager;
  assetManager.meshes.emplace("monkey", ws::loadOBJ(ws::ASSETS_FOLDER / "models/suzanne_smooth.obj"));
  assetManager.meshes.emplace("cube", ws::loadOBJ(ws::ASSETS_FOLDER / "models/cube.obj"));
  assetManager.meshes.emplace("sphere", ws::loadOBJ(ws::ASSETS_FOLDER / "models/sphere_ico.obj"));
  assetManager.textures.emplace("uv_grid", ws::ASSETS_FOLDER / "images/Wikipedia/UV_checker_Map_byValle.jpg");
  assetManager.textures.emplace("checkerboard", ws::ASSETS_FOLDER / "images/Wikipedia/checkerboard_pattern.png");
  assetManager.textures.emplace("wood", ws::ASSETS_FOLDER / "images/LearnOpenGL/container.jpg");
  assetManager.shaders.emplace("phong", ws::Shader{ws::ASSETS_FOLDER / "shaders/phong.vert", ws::ASSETS_FOLDER / "shaders/phong.frag"});
  assetManager.shaders.emplace("solid", ws::Shader{ws::ASSETS_FOLDER / "shaders/solid_color.vert", ws::ASSETS_FOLDER / "shaders/solid_color.frag"});
  assetManager.shaders.emplace("copy", ws::Shader{ws::ASSETS_FOLDER / "shaders/fullscreen_quad_without_vbo.vert", ws::ASSETS_FOLDER / "shaders/fullscreen_quad_texture_sampler.frag"});
  assetManager.shaders.emplace("lumi_tresh", ws::Shader{ws::ASSETS_FOLDER / "shaders/fullscreen_quad_without_vbo.vert", SRC / "lumi_tresh.frag"});
  assetManager.shaders.emplace("blur", ws::Shader{ws::ASSETS_FOLDER / "shaders/fullscreen_quad_without_vbo.vert", SRC / "blur.frag"});
  assetManager.materials.emplace("phongGround", ws::Material{
    .shader = assetManager.shaders.at("phong"),
    .parameters = {
      {"diffuseTexture", assetManager.textures.at("uv_grid")},
      {"specularTexture", assetManager.white}
    }
  });
  assetManager.materials.emplace("phongMonkey", ws::Material{
    .shader = assetManager.shaders.at("phong"), 
    .parameters = {
      {"diffuseTexture", assetManager.textures.at("checkerboard")},
      {"specularTexture", assetManager.white}
    }
  });
  assetManager.materials.emplace("solidSphere", ws::Material{
    .shader = assetManager.shaders.at("solid"), 
    .parameters = {
      {"u_Color", glm::vec4(1, 1, 0, 1)},
    }
  });
  assert(assetManager.doAllMaterialsHaveMatchingParametersAndUniforms());
  ws::Framebuffer sceneFbo{1, 1};
  ws::Framebuffer lumTreshFbo = ws::Framebuffer::makeDefaultColorOnly(1, 1);
  const int numMaxBlooms = 8;
  std::vector<ws::Framebuffer> bloomHorFbos; // Hmm... No need to reserve
  std::vector<ws::Framebuffer> bloomVerFbos; 
  for (int i = 0; i < numMaxBlooms; ++i) {
    std::vector<ws::Texture::Specs> colorSpecs = {{1, 1, ws::Texture::Format::RGBA8, ws::Texture::Filter::Linear, ws::Texture::Wrap::Repeat}};
    std::optional<ws::Texture::Specs> depthSpec = {};
    bloomHorFbos.emplace_back(colorSpecs, depthSpec);
    bloomVerFbos.emplace_back(colorSpecs, depthSpec);
  }
  ws::RenderableObject ground = {
    {"Ground", {glm::vec3{0, -1, 0}, glm::vec3{0, 0, 1}, 0, glm::vec3{20.f, .1f, 20.f}}},
    assetManager.meshes.at("cube"),
    assetManager.materials.at("phongGround"),
  };
  ws::RenderableObject monkey = {
      {"Monkey", {glm::vec3{0, -.15f, 0}, glm::vec3{1, 0, 0}, glm::radians(-30.f), glm::vec3{1.5f, 1.5f, 1.5f}}},
      assetManager.meshes.at("monkey"),
      assetManager.materials.at("phongMonkey"),
  };
  ws::RenderableObject sphere = {
      {"Sphere", {glm::vec3{1.6f, 0, 2.2f}, glm::vec3{0, 1, 0}, glm::radians(0.f), glm::vec3{1.f, 1.f, 1.f}}},
      assetManager.meshes.at("sphere"),
      assetManager.materials.at("solidSphere"),
  };
  ws::Scene scene{
    .renderables{ground, monkey, sphere},
    .directionalLights = std::vector<ws::DirectionalLight>{
      ws::DirectionalLight{
        .position = glm::vec3(1, 1, 1),
        .intensity = 0.5f,
        .direction = glm::vec3(-1, -1, -1),
        .color = glm::vec3(1, 1, 1),
      },
    },
  };
  ws::setParent(&ground, &scene.root);
  ws::setParent(&monkey, &scene.root);
  ws::setParent(&sphere, &scene.root);

  ws::AutoOrbitingCameraController orbitingCamController{scene.camera};
  orbitingCamController.radius = 10.f;
  orbitingCamController.theta = 0.3f;
  std::vector<std::reference_wrapper<ws::Texture>> texRefs{sceneFbo.getFirstColorAttachment(), lumTreshFbo.getFirstColorAttachment()};
  for (int n = 0; n < numMaxBlooms; ++n) {
    texRefs.push_back(bloomHorFbos[n].getFirstColorAttachment());  
    texRefs.push_back(bloomVerFbos[n].getFirstColorAttachment());  
  }
  ws::TextureViewer textureViewer{texRefs};
  ws::EditorWindow editorWindow{scene};
  ws::HierarchyWindow hierarchyWindow{scene};
  ws::InspectorWindow inspectorWindow{};
  workshop.shadersToReload = {assetManager.shaders.at("phong"), assetManager.shaders.at("solid"), assetManager.shaders.at("lumi_tresh"), assetManager.shaders.at("blur")};

  scene.ubo.compareSizeWithUniformBlock(assetManager.shaders.at("phong").getId(), "SceneUniforms"); 
   
  while (!workshop.shouldStop()) {
    workshop.beginFrame();
    const glm::uvec2 winSize = workshop.getWindowSize();
    sceneFbo.resizeIfNeeded(winSize.x, winSize.y);
    lumTreshFbo.resizeIfNeeded(winSize.x, winSize.y);
    int numBlooms = static_cast<int>(std::log2(std::min(winSize.x, winSize.y)));
    numBlooms = std::min(numBlooms, numMaxBlooms);
    for (const auto& [n, fbo] : bloomHorFbos | std::ranges::views::enumerate | std::ranges::views::take(numBlooms))
      fbo.resizeIfNeeded(static_cast<int>(winSize.x / std::pow(2, n + 1)), static_cast<int>(winSize.y / std::pow(2, n + 1)));
    for (const auto& [n, fbo] : bloomVerFbos | std::ranges::views::enumerate | std::ranges::views::take(numBlooms))
      fbo.resizeIfNeeded(static_cast<int>(winSize.x / std::pow(2, n + 1)), static_cast<int>(winSize.y / std::pow(2, n + 1)));


    ImGui::Begin("Bloom");
    static glm::vec3 bgColor{42 / 256.0, 96 / 256.0, 87 / 256.0};
    ImGui::ColorEdit3("BG Color", glm::value_ptr(bgColor));
    ImGui::Separator();
    static float luminanceThreshold = 0.75f;
    ImGui::SliderFloat("Luminance Treshold", &luminanceThreshold, 0, 1);
    ImGui::Text("Num Blooms %d", numBlooms);
    ImGui::End();

    orbitingCamController.update(workshop.getFrameDurationMs() * 0.001f);
    scene.camera.aspectRatio = static_cast<float>(winSize.x) / winSize.y;
    scene.uploadUniforms();

    glDisable(GL_BLEND);

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);
    sceneFbo.bind();
    glViewport(0, 0, winSize.x, winSize.y);
    glCullFace(GL_BACK);
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    glClearColor(bgColor.x, bgColor.y, bgColor.z, 1);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    scene.draw();
    sceneFbo.unbind();

    lumTreshFbo.bind();
    glViewport(0, 0, winSize.x, winSize.y);
    glClearColor(0, 0, 0, 0);
    glClear(GL_COLOR_BUFFER_BIT);
    sceneFbo.getFirstColorAttachment().bindToUnit(0);
    assetManager.shaders.at("lumi_tresh").setFloat("u_LuminanceThreshold", luminanceThreshold);
    assetManager.shaders.at("lumi_tresh").bind();
    assetManager.drawWithEmptyVao(6);
    ws::Shader::unbind();
    lumTreshFbo.unbind();

    for (int ix = 0; ix < numBlooms; ++ix) {
      bloomHorFbos[ix].bind();
      glViewport(0, 0, bloomHorFbos[ix].getFirstColorAttachment().specs.width, bloomHorFbos[ix].getFirstColorAttachment().specs.height);
      glClearColor(0, 0, 0, 0);
      glClear(GL_COLOR_BUFFER_BIT);

      assetManager.shaders.at("blur").setInteger("u_IsHorizontal", 0);
      (ix == 0 ? lumTreshFbo : bloomVerFbos[ix - 1]).getFirstColorAttachment().bindToUnit(0);
      assetManager.shaders.at("blur").bind();
      assetManager.drawWithEmptyVao(6);
      ws::Shader::unbind();
      bloomHorFbos[ix].unbind();

      bloomVerFbos[ix].bind();
      glViewport(0, 0, bloomVerFbos[ix].getFirstColorAttachment().specs.width, bloomVerFbos[ix].getFirstColorAttachment().specs.height);
      glClearColor(0, 0, 0, 0);
      glClear(GL_COLOR_BUFFER_BIT);

      assetManager.shaders.at("blur").setInteger("u_IsHorizontal", 1);
      bloomHorFbos[ix].getFirstColorAttachment().bindToUnit(0);
      assetManager.shaders.at("blur").bind();
      assetManager.drawWithEmptyVao(6);
      ws::Shader::unbind();
      bloomVerFbos[ix].unbind();
    }

    glViewport(0, 0, winSize.x, winSize.y);
    glClearColor(0, 0, 0, 0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);
    sceneFbo.getFirstColorAttachment().bindToUnit(0);
    assetManager.shaders.at("copy").bind();
    assetManager.drawWithEmptyVao(6);
    ws::Shader::unbind();


    glEnable(GL_BLEND);
    glBlendFunc(GL_ONE, GL_ONE);
    glDisable(GL_DEPTH_TEST);
    for (int ix = 0; ix < numBlooms; ++ix) {
      bloomVerFbos[ix].getFirstColorAttachment().bindToUnit(0);
      assetManager.shaders.at("copy").bind();
      assetManager.drawWithEmptyVao(6);
      ws::Shader::unbind();
    }

 	  workshop.drawUI();
    textureViewer.draw();
    static ws::VObjectPtr selectedObject;
    ws::VObjectPtr clickedObject = editorWindow.draw({}, selectedObject, workshop.getFrameDurationSec());
    selectedObject = hierarchyWindow.draw(clickedObject);
    inspectorWindow.inspectObject(selectedObject);

    workshop.endFrame();
  }

  std::println("Bye!");
  return 0;
}