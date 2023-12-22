#include <Workshop/Assets.hpp>
#include <Workshop/Camera.hpp>
#include <Workshop/Framebuffer.hpp>
#include <Workshop/Model.hpp>
#include <Workshop/Scene.hpp>
#include <Workshop/Shader.hpp>
#include <Workshop/Texture.hpp>
#include <Workshop/Transform.hpp>
#include <Workshop/UI.hpp>
#include <Workshop/Workshop.hpp>

#include <glad/gl.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/vec3.hpp>

import std.core;
import std.filesystem;

const std::filesystem::path SRC{SOURCE_DIR};

class AssetManager {
 public:
  std::unordered_map<std::string, ws::Mesh> meshes;
  std::unordered_map<std::string, ws::Texture> textures;
  std::unordered_map<std::string, ws::Framebuffer> framebuffers;
  std::unordered_map<std::string, ws::Shader> shaders;
};

struct DirectionalLight {
  glm::vec3 position{};
  glm::vec3 target{};
  // Shadow Parameters
  uint32_t shadowWidth = 1024;
  uint32_t shadowHeight = 1024;
  float side = 10.f;
  float near = 1.0f;
  float far = 7.5f;

  glm::mat4 getProjection() const {
    return glm::ortho(-side, side, -side, side, near, far);
  }

  glm::mat4 getView() const {
    return glm::lookAt(position, target, glm::vec3{0, 1, 0});
  }

  glm::mat4 getLightSpaceMatrix() const {
    return getProjection() * getView();
  }
};


int main() {
  std::println("Hi!");
  ws::Workshop workshop{2048, 1536, "Shadow Maps"};

  AssetManager assetManager;
  assetManager.meshes.emplace("cube", ws::Mesh{ws::loadOBJ(ws::ASSETS_FOLDER / "models/cube.obj")});
  assetManager.meshes.emplace("quad", ws::Mesh{ws::loadOBJ(ws::ASSETS_FOLDER / "models/quad.obj")});
  assetManager.textures.emplace("wood", ws::Texture{ws::ASSETS_FOLDER / "images/LearnOpenGL/container.jpg"});
  assetManager.shaders.emplace("unlit", ws::Shader{ws::ASSETS_FOLDER / "shaders/unlit.vert", ws::ASSETS_FOLDER / "shaders/unlit.frag"});
  assetManager.shaders.emplace("simpleDepth", ws::Shader{SRC / "shadow_mapping_depth.vert", SRC / "shadow_mapping_depth.frag"});
  // TODO: weirdly I need a move, can't emplace an FB directly
  auto fbo = ws::Framebuffer{};
  assetManager.framebuffers.emplace("shadowFBO", std::move(fbo));

  ws::RenderableObject ground = {
    ws::Object{std::string{"Ground"}, ws::Transform{glm::vec3{0, -0.5, 0}, glm::vec3{0, 0, 1}, 0, glm::vec3{25.f, 1, 25.f}}},
    assetManager.meshes.at("quad"),
    assetManager.shaders["unlit"],
    assetManager.textures["wood"],
  };
  ws::RenderableObject cube1 = {
    {"Cube1", {glm::vec3{0, 1.5f, 0}, glm::vec3{0, 0, 1}, 0, glm::vec3{1.f, 1.f, 1.f}}},
    assetManager.meshes.at("cube"),
    assetManager.shaders["unlit"],
    assetManager.textures["wood"],
  };
  ws::RenderableObject cube2 = {
    ws::Object{std::string{"Cube2"}, ws::Transform{glm::vec3{2.0f, 0.0f, 1.0f}, glm::vec3{0, 0, 1}, 0, glm::vec3{1.f, 1.f, 1.f}}},
    assetManager.meshes.at("cube"),
    assetManager.shaders["unlit"],
    assetManager.textures["wood"],
  };
  ws::RenderableObject cube3 = {
    ws::Object{std::string{"Cube3"}, ws::Transform{glm::vec3{-1.f, 0, 2.f}, glm::normalize(glm::vec3{1.f, 0, 1.f}), glm::radians(60.f), glm::vec3{.5f, .5f, .5f}}},
    assetManager.meshes.at("cube"),
    assetManager.shaders["unlit"],
    assetManager.textures["wood"],
  };
  ws::CameraObject cam1{
    ws::Object{std::string{"SceneCamera"}, ws::Transform{glm::vec3{0, 0, -5.f}, glm::vec3{0, 0, 1}, 0, glm::vec3{1, 1, 1}}},
  };
  ws::Scene scene{
    .renderables{ground, cube1, cube2, cube3},
    .cameras{cam1}
  };
  ws::setParent(&ground, &scene.root);
  ws::setParent(&cube1, &scene.root);
  ws::setParent(&cube2, &scene.root);
  ws::setParent(&cube3, &scene.root);
  ws::setParent(&cam1, &scene.root);

  ws::PerspectiveCamera3D& cam = scene.cameras[0].get().camera;
  ws::AutoOrbitingCamera3DViewController orbitingCamController{cam};
  orbitingCamController.radius = 7.7f;
  orbitingCamController.theta = 0.5;

  DirectionalLight light;
  light.position = {-2.f, 4.f, -1.f};
  assetManager.framebuffers.at("shadowFBO").resizeIfNeeded(light.shadowWidth, light.shadowHeight);

  const std::vector<std::reference_wrapper<ws::Texture>> texRefs{assetManager.framebuffers.at("shadowFBO").getFirstColorAttachment(), assetManager.framebuffers.at("shadowFBO").getDepthAttachment()};
  ws::TextureViewer textureViewer{texRefs};

  glEnable(GL_DEPTH_TEST);

  while (!workshop.shouldStop()) {
    workshop.beginFrame();
    const glm::uvec2 winSize = workshop.getWindowSize();

    workshop.imGuiDrawAppWindow();

    ImGui::Begin("Main");
    static glm::vec3 bgColor{144.f/255, 225.f/255, 236.f/255};
    ImGui::ColorEdit3("BG Color", glm::value_ptr(bgColor));
    ImGui::End();

    orbitingCamController.update(0.01f);
    cam.aspectRatio = static_cast<float>(winSize.x) / winSize.y;

    glClearColor(bgColor.x, bgColor.y, bgColor.z, 1);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    auto drawScene = [&]() {
      glViewport(0, 0, winSize.x, winSize.y);
      for (auto& renderable : scene.renderables) {
        renderable.get().shader.bind();
        renderable.get().mesh.bind();
        glBindTextureUnit(0, renderable.get().texture.getId());
        renderable.get().shader.setVector3("u_CameraPos", cam.getPosition());
        renderable.get().shader.setMatrix4("u_ViewFromWorld", cam.getViewFromWorld());
        renderable.get().shader.setMatrix4("u_ProjectionFromView", cam.getProjectionFromView());
        renderable.get().shader.setMatrix4("u_WorldFromObject", renderable.get().transform.getWorldFromObjectMatrix());
        renderable.get().mesh.draw();
        glBindTextureUnit(0, 0);
        renderable.get().mesh.unbind();
        renderable.get().shader.unbind();
      }
    };

    auto drawShadowMap = [&]() {
      glViewport(0, 0, light.shadowWidth, light.shadowHeight);
      assetManager.framebuffers.at("shadowFBO").bind();
      assetManager.shaders.at("simpleDepth").bind();
      glClearColor(0, 0, 0, 1);
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
      // cam.getProjectionFromView() * cam.getViewFromWorld() to see from camera's perspective
      assetManager.shaders.at("simpleDepth").setMatrix4("u_LightSpaceMatrix", light.getLightSpaceMatrix());

      for (auto& renderable : scene.renderables) {
        renderable.get().mesh.bind();
        assetManager.shaders.at("simpleDepth").setMatrix4("u_WorldFromObject", renderable.get().transform.getWorldFromObjectMatrix());
        renderable.get().mesh.draw();
        renderable.get().mesh.unbind();
      }

      assetManager.shaders.at("simpleDepth").unbind();
      assetManager.framebuffers.at("shadowFBO").unbind();
    };

    drawShadowMap();
    drawScene();

    textureViewer.draw();

    workshop.endFrame();
  }

  std::println("Bye!");
  return 0;
}