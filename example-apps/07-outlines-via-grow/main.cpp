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


  ws::Shader solidColorShader{ws::ASSETS_FOLDER / "shaders/solid_color.vert", ws::ASSETS_FOLDER / "shaders/solid_color.frag"};
  glm::vec4 outlineColor{0.85, 0.65, 0.15, 1};

  ws::Shader fullscreenShader{ws::ASSETS_FOLDER / "shaders/fullscreen_quad_without_vbo.vert", ws::ASSETS_FOLDER / "shaders/fullscreen_quad_texture_sampler.frag"};
  uint32_t fullscreenVao;
  glGenVertexArrays(1, &fullscreenVao);
  ws::Shader outlineShader{ws::ASSETS_FOLDER / "shaders/fullscreen_quad_without_vbo.vert", ws::ASSETS_FOLDER / "shaders/fullscreen_quad_outline.frag"};

  Renderable cube1{
    .mesh{ws::loadOBJ(ws::ASSETS_FOLDER / "models/suzanne.obj")},
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


  ws::Camera cam;
  ws::AutoOrbitingCameraController orbitingCamController{cam};
  orbitingCamController.radius = 13.8f;
  orbitingCamController.theta = 0.355f;

  ws::Framebuffer outlineA{};
  ws::Framebuffer outlineB{};

  const std::vector<std::reference_wrapper<ws::Texture>> texRefs{outlineA.getFirstColorAttachment(), outlineB.getFirstColorAttachment()};
  ws::TextureViewer textureViewer{texRefs};

  glEnable(GL_DEPTH_TEST);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);  // Needed for displaying textures with alpha < 1
  
  while (!workshop.shouldStop()) {
    workshop.beginFrame();
    const glm::uvec2 winSize = workshop.getWindowSize();
    outlineA.resizeIfNeeded(winSize.x, winSize.y);
    outlineB.resizeIfNeeded(winSize.x, winSize.y);
    glViewport(0, 0, winSize.x, winSize.y);

    workshop.drawUI();

    ImGui::Begin("Main");
    static glm::vec3 bgColor{42 / 256.0, 96 / 256.0, 87 / 256.0};
    ImGui::ColorEdit3("BG Color", glm::value_ptr(bgColor));
    ImGui::End();

    orbitingCamController.update(0.01f);
    cam.aspectRatio = static_cast<float>(winSize.x) / winSize.y;

    // Pass 1: Draw scene to screen
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

      ws::Texture::unbind();
      obj.mesh.unbind();
      obj.shader.unbind();
    }


    // Pass 2: Draw highlighted objects with solid color offscreen
    outlineA.bind();
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glDisable(GL_DEPTH_TEST);
    for (auto& objRef : renderables | std::views::filter(&Renderable::shouldHighlight)) {
      const auto& obj = objRef.get();
      solidColorShader.bind();
      obj.mesh.bind();

      solidColorShader.setMatrix4("u_WorldFromObject", obj.transform.getWorldFromObjectMatrix());
      solidColorShader.setMatrix4("u_ViewFromWorld", cam.getViewFromWorld());
      solidColorShader.setMatrix4("u_ProjectionFromView", cam.getProjectionFromView());
      solidColorShader.setVector4("u_Color", outlineColor);

      obj.mesh.draw();

      obj.mesh.unbind();
      solidColorShader.unbind();
    }
    glEnable(GL_DEPTH_TEST);
    outlineA.unbind();


    // Pass 3: Out-grow highlight solid color area
    outlineB.bind();
    glClearColor(0, 0, 0, 0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glDisable(GL_DEPTH_TEST);
    outlineShader.bind();
    glBindVertexArray(fullscreenVao);

    outlineA.getFirstColorAttachment().bind();
    glDrawArrays(GL_TRIANGLES, 0, 6);
    ws::Texture::unbind();

    glBindVertexArray(0);
    outlineShader.unbind();
    glEnable(GL_DEPTH_TEST);
    outlineB.unbind();


    // Pass 4: Draw highlights as overlay to screen
    glDisable(GL_DEPTH_TEST);
    fullscreenShader.bind();
    outlineB.getFirstColorAttachment().bind();

    glBindVertexArray(fullscreenVao);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    glBindVertexArray(0);

    ws::Texture::unbind();
    fullscreenShader.unbind();
    glEnable(GL_DEPTH_TEST);

    textureViewer.draw();

    workshop.endFrame();
  }

  std::println("Bye!");
  return 0;
}