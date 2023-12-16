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

struct RenderableObject;
struct CameraObject;
struct DummyObject;
//using RenderableObjectRef = std::reference_wrapper<RenderableObject>;
//using CameraObjectRef = std::reference_wrapper<CameraObject>;
//using VObject = std::variant<RenderableObjectRef, CameraObjectRef>;
using VObjectPtr = std::variant<DummyObject*, RenderableObject*, CameraObject*>;

class AssetManager {
 public:
  std::unordered_map<std::string, ws::Mesh> meshes;
  std::unordered_map<std::string, ws::Texture> textures;
  std::unordered_map<std::string, ws::Shader> shaders;
};

struct Object {
  std::string name;
  ws::Transform transform;

  VObjectPtr parent;
  std::unordered_set<VObjectPtr> children;
};

struct DummyObject : public Object {};

struct RenderableObject : public Object {
  ws::Mesh& mesh;
  ws::Shader& shader;
  ws::Texture& texture;
};

struct CameraObject : public Object {
  ws::PerspectiveCamera3D camera;
};

void setParent(VObjectPtr child, VObjectPtr parent1) {
  std::visit([&child](auto&& ptr) { ptr->children.insert(child); }, parent1);
  std::visit([&parent1](auto&& ptr) { ptr->parent = parent1; }, child);
}

class Scene {
 public:
  DummyObject root{"SceneRoot", {glm::vec3{0, 0, 0}, glm::vec3{0, 1, 0}, 0, glm::vec3{1, 1, 1}}};
  std::vector<RenderableObject> renderables;
  std::vector<CameraObject> cameras;
};

// TODO: move to Common.hpp
// helper type for the visitor #4
template <class... Ts>
struct overloaded : Ts... {
  using Ts::operator()...;
};
// explicit deduction guide (not needed as of C++20)
template <class... Ts>
overloaded(Ts...) -> overloaded<Ts...>;

// TODO: Make a more generic traversal function that'll take an `overloaded` and the root
void traverse(VObjectPtr node, int depth) {
  const bool isNull = std::visit([](auto&& ptr) { return ptr == nullptr; }, node);
  if (isNull)
    return;

  VObjectPtr parentNode = std::visit([](auto&& ptr) { return ptr->parent; }, node);
  const std::string& parentName = std::visit([](auto&& ptr) { return ptr != nullptr ? ptr->name : "NO_PARENT"; }, parentNode);

  std::visit(overloaded{
    [](auto arg) { throw "Unhandled VObjectPtr variant"; },
    [&](DummyObject* ptr) { 
      std::println("{} parent {} DummyObject name {}", depth, parentName, ptr->name);
    },
    [&](RenderableObject* ptr) { 
      RenderableObject& ref = *ptr;
      std::println("{} parent {} RenderableObject name {} verts {} tr.pos.x {}", depth, parentName, ref.name, ref.mesh.meshData.vertices.size(), ref.transform.position.x);
    },
    [&](CameraObject* ptr) { 
      CameraObject& ref = *ptr;
      std::println("{} parent {} CameraObject name {} verts fov {} tr.pos.x {}", depth, parentName, ref.name, ref.camera.fov, ref.transform.position.x);
    },
  }, node);

  const auto& children = std::visit([](auto&& ptr) { return ptr->children; }, node);
  for (auto childPtr : children)
    traverse(childPtr, depth + 1);
}


int main() {
  std::println("Hi!");
  ws::Workshop workshop{2048, 1536, "Shadow Maps"};

  AssetManager assetManager;
  assetManager.meshes.emplace("cube", ws::Mesh{ws::loadOBJ(ws::ASSETS_FOLDER / "models/cube.obj")});
  assetManager.meshes.emplace("quad", ws::Mesh{ws::loadOBJ(ws::ASSETS_FOLDER / "models/quad.obj")});
  assetManager.textures.emplace("wood", ws::Texture{ws::ASSETS_FOLDER / "images/LearnOpenGL/container.jpg"});
  assetManager.shaders.emplace("unlit", ws::Shader{ws::ASSETS_FOLDER / "shaders/unlit.vert", ws::ASSETS_FOLDER / "shaders/unlit.frag"});

  RenderableObject ground = {
    Object{std::string{"Ground"}, ws::Transform{glm::vec3{0, -0.5, 0}, glm::vec3{0, 0, 1}, 0, glm::vec3{25.f, 1, 25.f}}},
    assetManager.meshes.at("quad"),
    assetManager.shaders["unlit"],
    assetManager.textures["wood"],
  };
  RenderableObject cube1 = {
    {"Cube1", {glm::vec3{0, 1.5f, 0}, glm::vec3{0, 0, 1}, 0, glm::vec3{1.f, 1.f, 1.f}}},
    assetManager.meshes.at("cube"),
    assetManager.shaders["unlit"],
    assetManager.textures["wood"],
  };
  RenderableObject cube2 = {
    Object{std::string{"Cube2"}, ws::Transform{glm::vec3{2.0f, 0.0f, 1.0f}, glm::vec3{0, 0, 1}, 0, glm::vec3{1.f, 1.f, 1.f}}},
    assetManager.meshes.at("cube"),
    assetManager.shaders["unlit"],
    assetManager.textures["wood"],
  };
  RenderableObject cube3 = {
    Object{std::string{"Cube3"}, ws::Transform{glm::vec3{-1.f, 0, 2.f}, glm::normalize(glm::vec3{1.f, 0, 1.f}), glm::radians(60.f), glm::vec3{.5f, .5f, .5f}}},
    assetManager.meshes.at("cube"),
    assetManager.shaders["unlit"],
    assetManager.textures["wood"],
  };
  CameraObject cam1{
    Object{std::string{"SceneCamera"}, ws::Transform{glm::vec3{0, 0, -5.f}, glm::vec3{0, 0, 1}, 0, glm::vec3{1, 1, 1}}},
  };
  Scene scene{
    .renderables{ground, cube1, cube2, cube3},
    .cameras{cam1}
  };
  setParent(&ground, &scene.root);
  setParent(&cube1, &scene.root);
  setParent(&cube2, &scene.root);
  setParent(&cube3, &scene.root);
  setParent(&cam1, &scene.root);

  ws::PerspectiveCamera3D& cam = scene.cameras[0].camera;
  ws::AutoOrbitingCamera3DViewController orbitingCamController{cam};
  orbitingCamController.radius = 7.7f;
  orbitingCamController.theta = 0.5;

  std::println("TRAVERSING HIERARCHY TREE...");
  traverse(&scene.root, 0);

  std::vector<VObjectPtr> objects;
  std::ranges::transform(scene.renderables, std::back_inserter(objects), [](auto& obj) { return &obj; });
  std::ranges::transform(scene.cameras, std::back_inserter(objects), [](auto& obj) { return &obj; });
  std::println("ITERATING OVER ALL OBJECTS VECTOR...");
  for (auto objPtr : objects) {
    if (std::holds_alternative<RenderableObject*>(objPtr)) {
      RenderableObject& ref = *std::get<RenderableObject*>(objPtr);
      std::println("RenderableObject name {} verts {} tr.pos.x {}", ref.name, ref.mesh.meshData.vertices.size(), ref.transform.position.x);
    } else if (std::holds_alternative<CameraObject*>(objPtr)) {
      CameraObject& ref = *std::get<CameraObject*>(objPtr);
      std::println("CameraObject name {} verts fov {} tr.pos.x {}", ref.name, ref.camera.fov, ref.transform.position.x);
    }
    else
      throw "unknown object type";
  }

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

    for (auto& renderable : scene.renderables) {
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