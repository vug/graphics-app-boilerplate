#include "WorkshopTest.hpp"

#include <Workshop/Assets.hpp>
#include <Workshop/Model.hpp>
#include <Workshop/Scene.hpp>
#include <Workshop/Shader.hpp>
#include <Workshop/Texture.hpp>

#include <print>
#include <string>
#include <vector>

// To run all tests
// cd C:\Users\veliu\repos\graphics-app-boilerplate\out\build\x64-Debug\Workshop\test
// .\Tests.exe --gtest_filter=WorkshopTest.*

// .\Tests.exe --gtest_filter=WorkshopTest.HierarchyConstruction
TEST_F(WorkshopTest, HierarchyConstruction) {
  ws::Mesh cube{ws::loadOBJ(ws::ASSETS_FOLDER / "models/cube.obj")};
  ws::Mesh quad{ws::loadOBJ(ws::ASSETS_FOLDER / "models/quad.obj")};

  ws::Texture wood{ws::ASSETS_FOLDER / "images/LearnOpenGL/container.jpg"};
  ws::Shader unlit{ws::ASSETS_FOLDER / "shaders/unlit.vert", ws::ASSETS_FOLDER / "shaders/unlit.frag"};

  ws::RenderableObject ground = {
      ws::Object{std::string{"Ground"}, ws::Transform{glm::vec3{0, -0.5, 0}, glm::vec3{0, 0, 1}, 0, glm::vec3{25.f, 1, 25.f}}},
      quad, unlit, wood,
  };
  ws::RenderableObject cube1 = {
      {"Cube1", {glm::vec3{0, 1.5f, 0}, glm::vec3{0, 0, 1}, 0, glm::vec3{1.f, 1.f, 1.f}}},
      cube, unlit, wood
  };
  ws::RenderableObject cube2 = {
      ws::Object{std::string{"Cube2"}, ws::Transform{glm::vec3{2.0f, 0.0f, 1.0f}, glm::vec3{0, 0, 1}, 0, glm::vec3{1.f, 1.f, 1.f}}},
      cube, unlit, wood
  };
  ws::RenderableObject cube3 = {
      ws::Object{std::string{"Cube3"}, ws::Transform{glm::vec3{-1.f, 0, 2.f}, glm::normalize(glm::vec3{1.f, 0, 1.f}), glm::radians(60.f), glm::vec3{.5f, .5f, .5f}}},
      cube, unlit, wood
  };

  ws::Scene scene{
    .renderables{std::move(ground), std::move(cube1), std::move(cube2), std::move(cube3)},
  };

  ws::setParent(&scene.renderables[0], &scene.root);
  ws::setParent(&scene.renderables[1], &scene.renderables[0]);
  ws::setParent(&scene.renderables[2], &scene.renderables[0]);
  ws::setParent(&scene.renderables[3], &scene.renderables[1]);

  std::vector<std::string> objectNames;

  ws::traverse(&scene.root, 0, Overloaded{
    [&](ws::DummyObject* ptr) {
      objectNames.push_back(ptr->name);
    },
    [&](ws::RenderableObject* ptr) {
      ws::RenderableObject& ref = *ptr;
      objectNames.push_back(ptr->name);
    },
    [&](ws::CameraObject* ptr) {
      ws::CameraObject& ref = *ptr;
      objectNames.push_back(ptr->name);
    },
    [](auto arg) { throw "Unhandled VObjectPtr variant"; },
  });

  std::vector<std::string> expectedNames{"SceneRoot", "Ground", "Cube1", "Cube3", "Cube2"};
  ASSERT_EQ(objectNames, expectedNames);
}