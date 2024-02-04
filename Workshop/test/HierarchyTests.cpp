#include "WorkshopTest.hpp"

#include <Workshop/Assets.hpp>
#include <Workshop/Model.hpp>
#include <Workshop/Scene.hpp>
#include <Workshop/Shader.hpp>
#include <Workshop/Texture.hpp>

#include <print>
#include <ranges>
#include <string>
#include <variant>
#include <vector>

namespace rng = std::ranges;

#include <array>

void printMat4(const glm::mat4 mat) {
  std::print("{} {} {} {}\n{} {} {} {}\n{} {} {} {}\n{} {} {} {}\n",
               mat[0][0], mat[0][1], mat[0][2], mat[0][3],
               mat[1][0], mat[1][1], mat[1][2], mat[1][3],
               mat[2][0], mat[2][1], mat[2][2], mat[2][3],
               mat[3][0], mat[3][1], mat[3][2], mat[3][3]
  );
}

// To run all tests
// cd C:\Users\veliu\repos\graphics-app-boilerplate\out\build\x64-Debug\Workshop\test
// .\Tests.exe --gtest_filter=WorkshopTest.*

// .\Tests.exe --gtest_filter=WorkshopTest.HierarchyConstruction
TEST_F(WorkshopTest, HierarchyConstruction) {
  ws::Mesh cube{ws::loadOBJ(ws::ASSETS_FOLDER / "models/cube.obj")};
  ws::Mesh quad{ws::loadOBJ(ws::ASSETS_FOLDER / "models/quad.obj")};

  ws::Texture wood{ws::ASSETS_FOLDER / "images/LearnOpenGL/container.jpg"};
  ws::Texture metal{ws::ASSETS_FOLDER / "images/LearnOpenGL/metal.png"};
  ws::Shader unlit{ws::ASSETS_FOLDER / "shaders/unlit.vert", ws::ASSETS_FOLDER / "shaders/unlit.frag"};

  ws::RenderableObject ground = {
      ws::Object{std::string{"Ground"}, ws::Transform{glm::vec3{0, -0.5, 0}, glm::vec3{0, 0, 1}, 0, glm::vec3{25.f, 1, 25.f}}},
      quad,
      unlit,
      wood,
      metal,
  };
  ws::RenderableObject cube1 = {
      {"Cube1", {glm::vec3{0, 1.5f, 0}, glm::vec3{0, 0, 1}, 0, glm::vec3{1.f, 1.f, 1.f}}},
      cube,
      unlit,
      wood,
      metal,
  };
  ws::RenderableObject cube2 = {
      ws::Object{std::string{"Cube2"}, ws::Transform{glm::vec3{2.0f, 0.0f, 1.0f}, glm::vec3{0, 0, 1}, 0, glm::vec3{1.f, 1.f, 1.f}}},
      cube, unlit, wood, metal};
  ws::RenderableObject cube3 = {
      ws::Object{std::string{"Cube3"}, ws::Transform{glm::vec3{-1.f, 0, 2.f}, glm::normalize(glm::vec3{1.f, 0, 1.f}), glm::radians(60.f), glm::vec3{.5f, .5f, .5f}}},
      cube, unlit, wood, metal};
  ws::CameraObject camera{
      ws::Object{std::string{"MainCamera"}, ws::Transform{glm::vec3{0, 0, -5}, glm::normalize(glm::vec3{0, 1, 0}), 0, glm::vec3{1, 1, 1}}},
  };

  ws::Scene scene{
      .renderables{ground, cube1, cube2, cube3},
  };

  ws::setParent(&ground, &scene.root);
  ws::setParent(&cube1, &ground);
  ws::setParent(&cube2, &ground);
  ws::setParent(&cube3, &cube1);

  std::vector<std::string> objectNames;
  auto getObjectNames = [&objectNames](ws::VObjectPtr node, [[maybe_unused]] int depth) {
    // this would be easier to read/write w/o Overloaded. (See childName below). Keeping it here for demonstration only.
    std::visit(Overloaded{
                   [&](ws::DummyObject* ptr) {
                     objectNames.push_back(ptr->name);
                   },
                   [&](ws::RenderableObject* ptr) {
                     objectNames.push_back(ptr->name);
                   },
                   [&](ws::CameraObject* ptr) {
                     objectNames.push_back(ptr->name);
                   },
                   [](auto arg) { throw "Unhandled VObjectPtr variant"; },
               },
               node);
  };
  ws::traverse(&scene.root, 0, getObjectNames);
  std::vector<std::string> expectedObjectNames = {"SceneRoot", "Ground", "Cube1", "Cube3", "Cube2"};
  ASSERT_EQ(objectNames, expectedObjectNames);

  std::vector<std::tuple<int, std::string, std::string>> parentChildPairNames;
  auto getParentChildNames = [&parentChildPairNames](ws::VObjectPtr node, int depth) {
    ws::VObjectPtr parentNode = std::visit([](auto&& ptr) { return ptr->parent; }, node);
    const std::string& parentName = std::visit([](auto&& ptr) { return ptr != nullptr ? ptr->name : "NO_PARENT"; }, parentNode);
    const std::string& childName = std::visit([](auto&& ptr) { return ptr != nullptr ? ptr->name : "NO_CHILD"; }, node);
    parentChildPairNames.push_back({depth, parentName, childName});
  };
  ws::traverse(&scene.root, 0, getParentChildNames);

  std::vector<std::tuple<int, std::string, std::string>> expectedPairNames{
      {0, "NO_PARENT", "SceneRoot"},
      {1, "SceneRoot", "Ground"},
      {2, "Ground", "Cube1"},
      {3, "Cube1", "Cube3"},
      {2, "Ground", "Cube2"},
  };
  ASSERT_EQ(parentChildPairNames, expectedPairNames);

  std::vector<ws::VObjectPtr> allObjects{&ground, &cube1, &cube2, &cube3, &camera};
  std::vector<std::string> allObjectNames;
  std::vector<std::string> allObjectTypes;
  for (auto objPtr : allObjects) {
    const std::string name = std::visit([](auto&& ptr) { return ptr->name; }, objPtr);
    allObjectNames.push_back(name);
    // Unnecessary if condition but keeping it to demonstrate std::holds_alternative<>()
    if (std::holds_alternative<ws::RenderableObject*>(objPtr))
      allObjectTypes.push_back("RenderableObject");
    else if (std::holds_alternative<ws::CameraObject*>(objPtr))
      allObjectTypes.push_back("CameraObject");
    else
      allObjectTypes.push_back("UnknownObject");
  }

  std::vector<std::string> expectedAllObjectNames{"Ground", "Cube1", "Cube2", "Cube3", "MainCamera"};
  std::vector<std::string> expectedAllObjectTypes{"RenderableObject", "RenderableObject", "RenderableObject", "RenderableObject", "CameraObject"};
  ASSERT_EQ(allObjectNames, expectedAllObjectNames);

  //for (auto objPtr : allObjects) {
  //  const std::string& name = std::visit([](auto&& ptr) { return ptr->name; }, objPtr);

  //  const glm::mat4 localTransform = std::visit([](auto&& ptr) { return ptr->getLocalTransformMatrix(); }, objPtr);
  //  std::println("localTransform of {}:", name);
  //  printMat4(localTransform);

  //  const glm::mat4 globalTransform = std::visit([](auto&& ptr) { return ptr->getGlobalTransformMatrix(); }, objPtr);
  //  std::println("globalTransform of {}:", name);
  //  printMat4(globalTransform);
  //}

  ASSERT_EQ(ground.getLocalTransformMatrix(), ground.transform.getWorldFromObjectMatrix());
  ASSERT_EQ(ground.getGlobalTransformMatrix(), ground.transform.getWorldFromObjectMatrix());

  ASSERT_EQ(cube1.getLocalTransformMatrix(), cube1.transform.getWorldFromObjectMatrix());
  ASSERT_EQ(cube1.getGlobalTransformMatrix(), cube1.transform.getWorldFromObjectMatrix() * ground.transform.getWorldFromObjectMatrix());

  ASSERT_EQ(cube2.getLocalTransformMatrix(), cube2.transform.getWorldFromObjectMatrix());
  ASSERT_EQ(cube2.getGlobalTransformMatrix(), cube2.transform.getWorldFromObjectMatrix() * ground.transform.getWorldFromObjectMatrix());

  ASSERT_EQ(cube3.getLocalTransformMatrix(), cube3.transform.getWorldFromObjectMatrix());
  ASSERT_EQ(cube3.getGlobalTransformMatrix(), cube3.transform.getWorldFromObjectMatrix() * cube1.transform.getWorldFromObjectMatrix() * ground.transform.getWorldFromObjectMatrix());
}