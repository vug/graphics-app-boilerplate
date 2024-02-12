#include <Workshop/AssetManager.hpp>
#include <Workshop/Assets.hpp>
#include <Workshop/Camera.hpp>
#include <Workshop/Framebuffer.hpp>
#include <Workshop/Model.hpp>
#include <Workshop/Scene.hpp>
#include <Workshop/Shader.hpp>
#include <Workshop/Texture.hpp>
#include <Workshop/Transform.hpp>
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
  ws::Workshop workshop{2048, 1536, "Workshop App"};

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
  const ws::Shader outlineShader{ws::ASSETS_FOLDER / "shaders/solid_color.vert", ws::ASSETS_FOLDER / "shaders/solid_color.frag"};
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

  glm::vec4 outlineColor{0.85, 0.65, 0.15, 1};

  ws::AutoOrbitingCameraController orbitingCamController{scene.camera};
  orbitingCamController.radius = 13.8f;
  orbitingCamController.theta = 0.355f;
  orbitingCamController.speed = 0.5f;

  glEnable(GL_DEPTH_TEST);
  glEnable(GL_STENCIL_TEST);
  glStencilOp(GL_KEEP, GL_KEEP, GL_REPLACE);
  scene.ubo.compareSizeWithUniformBlock(assetManager.shaders.at("unlit").getId(), "SceneUniforms");
  
  while (!workshop.shouldStop()) {
    workshop.beginFrame();
    const glm::vec2 winSize = workshop.getWindowSize();

    workshop.drawUI();

    ImGui::Begin("Main");
    static glm::vec4 bgColor{42 / 256.f, 96 / 256.f, 87 / 256.f, 1.f};
    ImGui::ColorEdit4("BG Color", glm::value_ptr(bgColor));
    ImGui::End();

    orbitingCamController.update(0.01f);
    scene.camera.aspectRatio = static_cast<float>(winSize.x) / winSize.y;
    scene.uploadUniforms();

    ws::Framebuffer::clear(0, bgColor);
    ws::Framebuffer::clearStencil(0, 0);
    glEnable(GL_DEPTH_TEST);
    glStencilFunc(GL_ALWAYS, 1, 0xFF);
    for (const auto& [shouldHighlight, objRef] : std::views::zip(shouldHighlights, scene.renderables)) {
      const auto& obj = objRef.get();
      glStencilMask(shouldHighlight ? 0xFF : 0x00); // write into stencil buffer only for highlighted objects

      obj.material.uploadParameters();
      obj.material.shader.setMatrix4("u_WorldFromObject", obj.transform.getWorldFromObjectMatrix());
      obj.material.shader.bind();
      obj.mesh.draw();
    }
    ws::Shader::unbind();

    glStencilFunc(GL_NOTEQUAL, 1, 0xFF);
    glStencilMask(0x00);
    glDisable(GL_DEPTH_TEST);
    for (const auto& [shouldHighlight, objRef] : std::views::zip(shouldHighlights, scene.renderables)) {
      if (!shouldHighlight)
        continue;
      const auto& obj = objRef.get();
      const glm::mat4 scaledUp = glm::scale(obj.transform.getWorldFromObjectMatrix(), glm::vec3{1.1, 1.1, 1.1});
      outlineShader.setMatrix4("u_WorldFromObject", scaledUp);
      outlineShader.setMatrix4("u_ViewFromWorld", scene.camera.getViewFromWorld());
      outlineShader.setMatrix4("u_ProjectionFromView", scene.camera.getProjectionFromView());
      outlineShader.setVector4("u_Color", outlineColor);
      outlineShader.bind();
      obj.mesh.draw();
    }
    ws::Shader::unbind();
    glStencilMask(0xFF);
    glEnable(GL_DEPTH_TEST);

    workshop.endFrame();
  }

  std::println("Bye!");
  return 0;
}