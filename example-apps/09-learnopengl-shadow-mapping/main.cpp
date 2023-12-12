#include <Workshop/Assets.hpp>
#include <Workshop/Camera.hpp>
#include <Workshop/Model.hpp>
#include <Workshop/Shader.hpp>
#include <Workshop/Texture.hpp>
#include <Workshop/Transform.hpp>
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

class AssetManager {
 public:
  std::unordered_map<std::string, ws::Mesh> meshes;
  std::unordered_map<std::string, ws::Texture> textures;
  std::unordered_map<std::string, ws::Shader> shaders;
};

struct Renderable {
  std::string name;
  ws::Transform transform;
  ws::Mesh& mesh;
  ws::Shader& shader;
  ws::Texture& texture;
};
using Scene = std::vector<Renderable>;

int main() {
  std::println("Hi!");
  ws::Workshop workshop{2048, 1536, "Shadow Maps"};
  AssetManager assetManager;
  assetManager.meshes.emplace("cube", ws::Mesh{ws::loadOBJ(ws::ASSETS_FOLDER / "models/cube.obj")});
  assetManager.meshes.emplace("quad", ws::Mesh{ws::loadOBJ(ws::ASSETS_FOLDER / "models/quad.obj")});
  assetManager.textures.emplace("wood", ws::Texture{ws::ASSETS_FOLDER / "images/LearnOpenGL/container.jpg"});
  assetManager.shaders.emplace("unlit", ws::Shader{ws::ASSETS_FOLDER / "shaders/unlit.vert", ws::ASSETS_FOLDER / "shaders/unlit.frag"});

  Renderable ground = {
      .name = "ground",
      .transform{glm::vec3{0, -0.5, 0}, glm::vec3{0, 0, 1}, 0, glm::vec3{25.f, 1, 25.f}},
      .mesh{assetManager.meshes.at("quad")},
      .shader = assetManager.shaders["unlit"],
      .texture = assetManager.textures["wood"],
  };
  Renderable cube1 = {
      .name = "cube1",
      .transform{glm::vec3{0, 1.5f, 0}, glm::vec3{0, 0, 1}, 0, glm::vec3{1.f, 1.f, 1.f}},
      .mesh{assetManager.meshes.at("cube")},
      .shader = assetManager.shaders["unlit"],
      .texture = assetManager.textures["wood"],
  };
  Renderable cube2 = {
      .name = "cube2",
      .transform{glm::vec3{2.0f, 0.0f, 1.0f}, glm::vec3{0, 0, 1}, 0, glm::vec3{1.f, 1.f, 1.f}},
      .mesh{assetManager.meshes.at("cube")},
      .shader = assetManager.shaders["unlit"],
      .texture = assetManager.textures["wood"],
  };
  Renderable cube3 = {
      .name = "cube3",
      .transform{glm::vec3{-1.f, 0, 2.f}, glm::normalize(glm::vec3{1.f, 0, 1.f}), glm::radians(60.f), glm::vec3{.5f, .5f, .5f}},
      .mesh{assetManager.meshes.at("cube")},
      .shader = assetManager.shaders["unlit"],
      .texture = assetManager.textures["wood"],
  };
  Scene scene{ground, cube1, cube2, cube3,};

  ws::PerspectiveCamera3D cam;
  ws::AutoOrbitingCamera3DViewController orbitingCamController{cam};
  orbitingCamController.radius = 7.7f;
  orbitingCamController.theta = 0.5;

  glEnable(GL_DEPTH_TEST);

  while (!workshop.shouldStop()) {
    workshop.beginFrame();
    const glm::uvec2 winSize = workshop.getWindowSize();
    glViewport(0, 0, winSize.x, winSize.y);

    workshop.imGuiDrawAppWindow();

    ImGui::Begin("Main");
    static glm::vec3 bgColor{144.f/255, 225.f/255, 236.f/255};
    ImGui::ColorEdit3("BG Color", glm::value_ptr(bgColor));
    ImGui::End();

    orbitingCamController.update(0.01f);
    cam.aspectRatio = static_cast<float>(winSize.x) / winSize.y;

    glClearColor(bgColor.x, bgColor.y, bgColor.z, 1);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    for (auto& renderable : scene) {
      renderable.shader.bind();
      renderable.mesh.bind();
      glBindTextureUnit(0, renderable.texture.getId());
      renderable.shader.setVector3("u_CameraPos", cam.getPosition());
      renderable.shader.setMatrix4("u_ViewFromWorld", cam.getViewFromWorld());
      renderable.shader.setMatrix4("u_ProjectionFromView", cam.getProjectionFromView());
      renderable.shader.setMatrix4("u_WorldFromObject", renderable.transform.getWorldFromObjectMatrix());
      renderable.mesh.draw();
      glBindTextureUnit(0, 0);
      renderable.mesh.unbind();
      renderable.shader.unbind();
    }

    workshop.endFrame();
  }

  return 0;
}