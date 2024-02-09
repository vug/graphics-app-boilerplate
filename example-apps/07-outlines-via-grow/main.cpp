#include <Workshop/AssetManager.hpp>
#include <Workshop/Assets.hpp>
#include <Workshop/Camera.hpp>
#include <Workshop/Framebuffer.hpp>
#include <Workshop/Model.hpp>
#include <Workshop/Shader.hpp>
#include <Workshop/Texture.hpp>
#include <Workshop/Transform.hpp>
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

int main() {
  std::println("Hi!");
  ws::Workshop workshop{2048, 1536, "Outlines via Growth"};

  ws::AssetManager assetManager;
  assetManager.meshes.emplace("suzanne", ws::loadOBJ(ws::ASSETS_FOLDER / "models/suzanne.obj"));
  assetManager.meshes.emplace("cube", ws::loadOBJ(ws::ASSETS_FOLDER / "models/cube.obj"));
  assetManager.meshes.emplace("quad", ws::loadOBJ(ws::ASSETS_FOLDER / "models/quad.obj"));
  assetManager.textures.emplace("marble", ws::Texture{ws::ASSETS_FOLDER / "images/LearnOpenGL/marble.jpg"});
  assetManager.textures.emplace("container", ws::Texture{ws::ASSETS_FOLDER / "images/LearnOpenGL/container.jpg"});
  assetManager.textures.emplace("metal", ws::Texture{ws::ASSETS_FOLDER / "images/LearnOpenGL/metal.png"});
  assetManager.shaders.emplace("unlit", ws::Shader{ws::ASSETS_FOLDER / "shaders/unlit.vert", ws::ASSETS_FOLDER / "shaders/unlit.frag"});
  assetManager.shaders.emplace("solid_color", ws::Shader{ws::ASSETS_FOLDER / "shaders/editor/solid_color.vert", ws::ASSETS_FOLDER / "shaders/editor/solid_color.frag"});
  assetManager.shaders.emplace("fullscreen", ws::Shader{ws::ASSETS_FOLDER / "shaders/fullscreen_quad_without_vbo.vert", ws::ASSETS_FOLDER / "shaders/fullscreen_quad_texture_sampler.frag"});
  const ws::Shader outlineShader{ws::ASSETS_FOLDER / "shaders/fullscreen_quad_without_vbo.vert", ws::ASSETS_FOLDER / "shaders/fullscreen_quad_outline.frag"};
  assetManager.materials.emplace("unlit-monkey", ws::Material{
    .shader = assetManager.shaders.at("unlit"),
    .parameters = {
      {"mainTex", assetManager.textures.at("marble")},
      {"u_Color", glm::vec4(1, 1, 1, 1)},
    },
  });
  assetManager.materials.emplace("unlit-cube", ws::Material{
    .shader = assetManager.shaders.at("unlit"),
    .parameters = {
      {"mainTex", assetManager.textures.at("container")},
      {"u_Color", glm::vec4(1, 1, 1, 1)},
    },
  });
  assetManager.materials.emplace("unlit-quad", ws::Material{
    .shader = assetManager.shaders.at("unlit"),
    .parameters = {
      {"mainTex", assetManager.textures.at("metal")},
      {"u_Color", glm::vec4(1, 1, 1, 1)},
    },
  });
  assert(assetManager.doAllMaterialsHaveMatchingParametersAndUniforms());

  ws::RenderableObject monkey = {
    {"Monkey", {glm::vec3{-1, 0.5, -1}, glm::vec3{0, 0, 1}, 0, glm::vec3{1, 1, 1}}},
    assetManager.meshes.at("suzanne"),
    assetManager.materials.at("unlit-monkey"),
  };
  ws::RenderableObject cube = {
    {"Cube", {glm::vec3{2, 0.5, 0}, glm::vec3{0, 0, 1}, 0, glm::vec3{1, 1, 1}}},
    assetManager.meshes.at("cube"),
    assetManager.materials.at("unlit-cube"),
  };
  ws::RenderableObject quad = {
    {"Quad", {glm::vec3{0, 0, 0}, glm::vec3{0, 0, 1}, 0, glm::vec3{10, 0, 10}}},
    assetManager.meshes.at("cube"),
    assetManager.materials.at("unlit-quad"),
  };

  ws::Scene scene{
    .renderables = {monkey, cube, quad},
  };
  std::vector<bool> shouldHighlights = {true, true, false};

  ws::AutoOrbitingCameraController orbitingCamController{scene.camera};
  orbitingCamController.radius = 13.8f;
  orbitingCamController.theta = 0.355f;
  orbitingCamController.speed = 0.5f;

  ws::Framebuffer outlineA{};
  ws::Framebuffer outlineB{};

  const std::vector<std::reference_wrapper<ws::Texture>> texRefs{outlineA.getFirstColorAttachment(), outlineB.getFirstColorAttachment()};
  ws::TextureViewer textureViewer{texRefs};

  glEnable(GL_DEPTH_TEST);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);  // Needed for displaying textures with alpha < 1
  scene.ubo.compareSizeWithUniformBlock(assetManager.shaders.at("unlit").getId(), "SceneUniforms");
  
  while (!workshop.shouldStop()) {
    workshop.beginFrame();
    const glm::uvec2 winSize = workshop.getWindowSize();
    outlineA.resizeIfNeeded(winSize.x, winSize.y);
    outlineB.resizeIfNeeded(winSize.x, winSize.y);
    glViewport(0, 0, winSize.x, winSize.y);

    workshop.drawUI();

    ImGui::Begin("Outlines via Growth");
    static glm::vec3 bgColor{42 / 256.0, 96 / 256.0, 87 / 256.0};
    ImGui::ColorEdit3("BG Color", glm::value_ptr(bgColor));
    static glm::vec4 outlineColor{0.85, 0.65, 0.15, 1};
    ImGui::ColorEdit4("Outline Color", glm::value_ptr(outlineColor));
    ImGui::End();

    orbitingCamController.update(0.01f);
    scene.camera.aspectRatio = static_cast<float>(winSize.x) / winSize.y;
    scene.uploadUniforms();

    // Pass 1: Draw scene to screen
    glClearColor(bgColor.x, bgColor.y, bgColor.z, 1);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    scene.draw();

    // Pass 2: Draw highlighted objects with solid color offscreen
    outlineA.bind();
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glDisable(GL_DEPTH_TEST);
    for (const auto& [shouldHighlight, objRef] : std::views::zip(shouldHighlights, scene.renderables)) {
      if (!shouldHighlight)
        continue;

      const auto& obj = objRef.get();
      const auto& shader = assetManager.shaders.at("solid_color");
      shader.setMatrix4("u_WorldFromObject", obj.transform.getWorldFromObjectMatrix());
      shader.setMatrix4("u_ViewFromWorld", scene.camera.getViewFromWorld());
      shader.setMatrix4("u_ProjectionFromView", scene.camera.getProjectionFromView());
      shader.setVector4("u_Color", outlineColor);
      shader.bind();
      obj.mesh.draw();
      shader.unbind();
    }
    glEnable(GL_DEPTH_TEST);
    outlineA.unbind();


    // Pass 3: Out-grow highlight solid color area
    glDisable(GL_DEPTH_TEST);
    outlineB.bind();
    glClearColor(0, 0, 0, 0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    outlineShader.setVector4("u_OutlineColor", outlineColor);
    outlineA.getFirstColorAttachment().bindToUnit(0);
    outlineShader.bind();
    assetManager.drawWithEmptyVao(6);
    outlineShader.unbind();
    outlineB.unbind();
    glEnable(GL_DEPTH_TEST);


    // Pass 4: Draw highlights as overlay to screen
    glDisable(GL_DEPTH_TEST);
    outlineB.getFirstColorAttachment().bindToUnit(0);
    assetManager.shaders.at("fullscreen").bind();
    assetManager.drawWithEmptyVao(6);
    assetManager.shaders.at("fullscreen").unbind();
    glEnable(GL_DEPTH_TEST);

    textureViewer.draw();

    workshop.endFrame();
  }

  std::println("Bye!");
  return 0;
}