#include <Workshop/AssetManager.hpp>
#include <Workshop/Assets.hpp>
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
  ws::Workshop workshop{1920, 1080, "Boilerplate app"};

  ws::AssetManager assetManager;
  assetManager.meshes.emplace("monkey", ws::loadOBJ(ws::ASSETS_FOLDER / "models/suzanne.obj"));
  assetManager.meshes.emplace("cube", ws::loadOBJ(ws::ASSETS_FOLDER / "models/cube.obj"));
  assetManager.textures.emplace("uv_grid", ws::ASSETS_FOLDER / "images/Wikipedia/UV_checker_Map_byValle.jpg");
  assetManager.textures.emplace("checkerboard", ws::ASSETS_FOLDER / "images/Wikipedia/checkerboard_pattern.png");
  assetManager.textures.emplace("wood", ws::ASSETS_FOLDER / "images/LearnOpenGL/container.jpg");
  assetManager.shaders.emplace("phong", ws::Shader{ws::ASSETS_FOLDER / "shaders/phong.vert", ws::ASSETS_FOLDER / "shaders/phong.frag"});
  assetManager.shaders.emplace("unlit", ws::Shader{ws::ASSETS_FOLDER / "shaders/unlit.vert", ws::ASSETS_FOLDER / "shaders/unlit.frag"});
  assetManager.shaders.emplace("boilerplate", ws::Shader{SRC / "boilerplate.vert", SRC / "boilerplate.frag"});
  assetManager.materials.emplace("phongGround", ws::Material{
    .shader = assetManager.shaders.at("phong"),
    .parameters = {
      {"diffuseTexture", assetManager.textures.at("uv_grid")},
      {"specularTexture", assetManager.white}
    }
  });
  assetManager.materials.emplace("unlit-monkey", ws::Material{
    .shader = assetManager.shaders.at("unlit"),
    .parameters = {
      {"mainTex", assetManager.textures.at("checkerboard")},
      {"u_Color", glm::vec4(1, 1, 1, 1)},
    }
  });
  assetManager.materials.emplace("boilerplate-box", ws::Material{
    .shader = assetManager.shaders.at("boilerplate"),
    .parameters = {
      {"mainTex", assetManager.textures.at("wood")},
      {"secondTex", assetManager.textures.at("checkerboard")},
    }
  });
  assert(assetManager.doAllMaterialsHaveMatchingParametersAndUniforms());
  ws::Framebuffer offscreenFbo = ws::Framebuffer::makeDefaultColorOnly(1, 1);

  ws::RenderableObject ground = {
      {"Ground", {glm::vec3{0, -1, 0}, glm::vec3{0, 0, 1}, 0, glm::vec3{20.f, .1f, 20.f}}},
      assetManager.meshes.at("cube"),
      assetManager.materials.at("phongGround"),
  };
  ws::RenderableObject monkey = {
      {"Monkey", {glm::vec3{0, -.15f, 0}, glm::vec3{1, 0, 0}, glm::radians(-30.f), glm::vec3{1.5f, 1.5f, 1.5f}}},
      assetManager.meshes.at("monkey"),
      assetManager.materials.at("unlit-monkey"),
  };
  ws::RenderableObject box = {
      {"Box", {glm::vec3{1.6f, 0, 2.2f}, glm::vec3{0, 1, 0}, glm::radians(-22.f), glm::vec3{1.f, 2.f, 2.f}}},
      assetManager.meshes.at("cube"),
      assetManager.materials.at("boilerplate-box"),
  };
  ws::Scene scene{
    .renderables{ground, monkey, box},
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
  ws::setParent(&box, &scene.root);

  ws::AutoOrbitingCameraController orbitingCamController{scene.camera};
  orbitingCamController.radius = 10.f;
  orbitingCamController.theta = 0.3f;
  const std::vector<std::reference_wrapper<ws::Texture>> texRefs{offscreenFbo.getFirstColorAttachment()};
  ws::TextureViewer textureViewer{texRefs};
  ws::EditorWindow editorWindow{scene};
  ws::HierarchyWindow hierarchyWindow{scene};
  ws::InspectorWindow inspectorWindow{};
  workshop.shadersToReload = {assetManager.shaders.at("phong"), assetManager.shaders.at("unlit"), assetManager.shaders.at("boilerplate")};
  
  glEnable(GL_DEPTH_TEST);
  scene.ubo.compareSizeWithUniformBlock(assetManager.shaders.at("phong").getId(), "SceneUniforms");
  
  while (!workshop.shouldStop()) {
    workshop.beginFrame();
    const glm::uvec2 winSize = workshop.getWindowSize();
    offscreenFbo.resizeIfNeeded(winSize.x, winSize.y); // can be resized by something else

    ImGui::Begin("Boilerplate");
    static glm::vec4 bgColor{42 / 256.f, 96 / 256.f, 87 / 256.f, 1.f};
    ImGui::ColorEdit4("BG Color", glm::value_ptr(bgColor));
    ImGui::Separator();
    ImGui::End();

    orbitingCamController.update(workshop.getFrameDurationMs() * 0.001f);
    scene.camera.aspectRatio = static_cast<float>(winSize.x) / winSize.y;
    scene.uploadUniforms();

    offscreenFbo.bind();
    glViewport(0, 0, winSize.x, winSize.y);
    glDisable(GL_CULL_FACE);
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    offscreenFbo.clearColor({1, 1, 1, 0});
    scene.draw(assetManager.materials.at("boilerplate-box"));
    ws::Framebuffer::unbind();

    glViewport(0, 0, winSize.x, winSize.y);
    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    ws::Framebuffer::clear(0, bgColor);
    scene.draw();

 	  workshop.drawUI();
    textureViewer.draw();
    static ws::VObjectPtr selectedObject;
    ws::VObjectPtr clickedObject = editorWindow.draw({} , selectedObject, workshop.getFrameDurationSec());
    selectedObject = hierarchyWindow.draw(clickedObject);
    inspectorWindow.inspectObject(selectedObject);

    workshop.endFrame();
  }

  std::println("Bye!");
  return 0;
}