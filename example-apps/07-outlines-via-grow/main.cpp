#include <Workshop/Assets.hpp>
#include <Workshop/Camera.hpp>
#include <Workshop/Framebuffer.hpp>
#include <Workshop/Model.hpp>
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

struct Renderable {
  ws::Mesh mesh;
  ws::Shader shader;
  ws::Transform transform;
  glm::vec4 color{1, 1, 1, 1};
  ws::Texture texture;
  bool shouldHighlight{false};
};

int main()
{
  std::println("Hi!");
  ws::Workshop workshop{2048, 1536, "Workshop App"};


  ws::Shader outlineShader{ws::ASSETS_FOLDER / "shaders/solid_color.vert", ws::ASSETS_FOLDER / "shaders/solid_color.frag"};
  glm::vec4 outlineColor{0.85, 0.65, 0.15, 1};
  Renderable cube1{
    .mesh{ws::loadOBJ(ws::ASSETS_FOLDER / "models/cube.obj")},
    .shader{ws::ASSETS_FOLDER / "shaders/unlit.vert", ws::ASSETS_FOLDER / "shaders/unlit.frag"},
    .transform{glm::vec3{-1, 0.5, -1}, glm::vec3{0, 0, 1}, 0, glm::vec3{1, 1, 1}},
    //.color{0.9, 0.8, 0.1, 1.0},
    .texture{ws::ASSETS_FOLDER / "images/LearnOpenGL/marble.jpg"},
    .shouldHighlight{true},
  };
  Renderable cube2{
    .mesh{ws::loadOBJ(ws::ASSETS_FOLDER / "models/cube.obj")},
    .shader{ws::ASSETS_FOLDER / "shaders/unlit.vert", ws::ASSETS_FOLDER / "shaders/unlit.frag"},
    .transform{glm::vec3{2, 0.5, 0}, glm::vec3{0, 0, 1}, 0, glm::vec3{1, 1, 1}},
    //.color{0.9, 0.1, 0.8, 1.0},
    .texture{ws::ASSETS_FOLDER / "images/LearnOpenGL/marble.jpg"},
    .shouldHighlight{true},
  };
  Renderable quad{
    .mesh{ws::loadOBJ(ws::ASSETS_FOLDER / "models/quad.obj")},
    .shader{ws::ASSETS_FOLDER / "shaders/unlit.vert", ws::ASSETS_FOLDER / "shaders/unlit.frag"},
    .transform{glm::vec3{0, 0, 0}, glm::vec3{0, 0, 1}, 0, glm::vec3{10, 0, 10}},
    //.color{0.1, 0.9, 0.8, 1.0},
    .texture{ws::ASSETS_FOLDER / "images/LearnOpenGL/metal.png"},
  };
  std::vector<std::reference_wrapper<Renderable>> renderables = {cube1, quad, cube2};

  ws::PerspectiveCamera3D cam;
  ws::AutoOrbitingCamera3DViewController orbitingCamController{cam};
  orbitingCamController.radius = 13.8f;
  orbitingCamController.theta = 0.355f;

  ws::Framebuffer outlineA{};
  ws::Framebuffer outlineB{};

  glEnable(GL_DEPTH_TEST);
  
  while (!workshop.shouldStop()) {
    workshop.beginFrame();
    const glm::uvec2 winSize = workshop.getWindowSize();
    outlineA.resizeIfNeeded(winSize.x, winSize.y);
    outlineB.resizeIfNeeded(winSize.x, winSize.y);

    ImGui::Begin("Main");
    static bool shouldShowImGuiDemo = false;
    ImGui::Checkbox("Show Demo", &shouldShowImGuiDemo);
    if (shouldShowImGuiDemo)
      ImGui::ShowDemoWindow();

    static glm::vec3 bgColor{42 / 256.0, 96 / 256.0, 87 / 256.0};
    ImGui::ColorEdit3("BG Color", glm::value_ptr(bgColor));
    ImGui::End();

    orbitingCamController.update(0.01f);
    cam.aspectRatio = static_cast<float>(winSize.x) / winSize.y;

    glClearColor(bgColor.x, bgColor.y, bgColor.z, 1);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    for (auto& objRef : renderables) {
      const auto& obj = objRef.get();

      obj.shader.bind();
      obj.mesh.bind();
      obj.texture.bind();

      obj.shader.setMatrix4("u_WorldFromObject", obj.transform.getWorldFromObjectMatrix());
      obj.shader.setMatrix4("u_ViewFromWorld", cam.getViewFromWorld());
      obj.shader.setMatrix4("u_ProjectionFromView", cam.getProjectionFromView());
      obj.shader.setVector4("u_Color", obj.color);

      obj.mesh.draw();

      obj.texture.unbind();
      obj.mesh.unbind();
      obj.shader.unbind();
    }

    outlineA.bind();
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glDisable(GL_DEPTH_TEST);
    for (auto& objRef : renderables | std::views::filter(&Renderable::shouldHighlight)) {
      const auto& obj = objRef.get();
      outlineShader.bind();
      obj.mesh.bind();

      outlineShader.setMatrix4("u_WorldFromObject", obj.transform.getWorldFromObjectMatrix());
      outlineShader.setMatrix4("u_ViewFromWorld", cam.getViewFromWorld());
      outlineShader.setMatrix4("u_ProjectionFromView", cam.getProjectionFromView());
      outlineShader.setVector4("u_Color", outlineColor);

      obj.mesh.draw();

      obj.mesh.unbind();
      outlineShader.unbind();
    }
    glEnable(GL_DEPTH_TEST);
    outlineA.unbind();

    ImGui::Begin("Texture Viewer");
    const auto& tex = outlineA.getFirstColorAttachment();
    ImGui::Image((void*)(intptr_t)tex.getId(), ImVec2{tex.specs.width / 4.f, tex.specs.height / 4.f}, {0, 1}, {1, 0});
    ImGui::End();

    workshop.endFrame();
  }

  std::println("Bye!");
  return 0;
}