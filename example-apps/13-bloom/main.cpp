#include <Workshop/Assets.hpp>
#include <Workshop/AssetManager.hpp>
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
  ws::Workshop workshop{1920, 1080, "Bloom"};

  ws::AssetManager assetManager;
  assetManager.meshes.emplace("monkey", ws::loadOBJ(ws::ASSETS_FOLDER / "models/suzanne.obj"));
  assetManager.meshes.emplace("cube", ws::loadOBJ(ws::ASSETS_FOLDER / "models/cube.obj"));
  assetManager.meshes.emplace("sphere", ws::loadOBJ(ws::ASSETS_FOLDER / "models/sphere_ico.obj"));
  assetManager.textures.emplace("uv_grid", ws::ASSETS_FOLDER / "images/Wikipedia/UV_checker_Map_byValle.jpg");
  assetManager.textures.emplace("checkerboard", ws::ASSETS_FOLDER / "images/Wikipedia/checkerboard_pattern.png");
  assetManager.textures.emplace("wood", ws::ASSETS_FOLDER / "images/LearnOpenGL/container.jpg");
  assetManager.shaders.emplace("phong", ws::Shader{ws::ASSETS_FOLDER / "shaders/phong.vert", ws::ASSETS_FOLDER / "shaders/phong.frag"});
  assetManager.shaders.emplace("solid", ws::Shader{ws::ASSETS_FOLDER / "shaders/solid_color.vert", ws::ASSETS_FOLDER / "shaders/solid_color.frag"});
  assetManager.shaders.emplace("copy", ws::Shader{ws::ASSETS_FOLDER / "shaders/fullscreen_quad_without_vbo.vert", ws::ASSETS_FOLDER / "shaders/fullscreen_quad_texture_sampler.frag"});
  assetManager.shaders.emplace("boilerplate", ws::Shader{SRC / "boilerplate.vert", SRC / "boilerplate.frag"});
  ws::Shader debugShader{ws::ASSETS_FOLDER / "shaders/debug.vert", ws::ASSETS_FOLDER / "shaders/debug.frag"};
  ws::Framebuffer offscreenFbo = ws::Framebuffer(1, 1);

  ws::RenderableObject ground = {
      {"Ground", {glm::vec3{0, -1, 0}, glm::vec3{0, 0, 1}, 0, glm::vec3{20.f, .1f, 20.f}}},
      assetManager.meshes.at("cube"),
      assetManager.shaders.at("phong"),
      assetManager.textures.at("uv_grid"),
      assetManager.white,
  };
  ws::RenderableObject monkey = {
      {"Monkey", {glm::vec3{0, -.15f, 0}, glm::vec3{1, 0, 0}, glm::radians(-30.f), glm::vec3{1.5f, 1.5f, 1.5f}}},
      assetManager.meshes.at("monkey"),
      assetManager.shaders.at("phong"),
      assetManager.textures.at("checkerboard"),
      assetManager.white,
  };
  ws::RenderableObject sphere = {
      {"Sphere", {glm::vec3{1.6f, 0, 2.2f}, glm::vec3{0, 1, 0}, glm::radians(0.f), glm::vec3{1.f, 1.f, 1.f}}},
      assetManager.meshes.at("sphere"),
      assetManager.shaders.at("solid"),
      assetManager.textures.at("wood"),
      assetManager.textures.at("checkerboard"),
  };
  ws::Camera cam;
  ws::Scene scene{
    .renderables{ground, monkey, sphere},
  };
  ws::setParent(&ground, &scene.root);
  ws::setParent(&monkey, &scene.root);
  ws::setParent(&sphere, &scene.root);

  ws::AutoOrbitingCameraController orbitingCamController{cam};
  orbitingCamController.radius = 10.f;
  orbitingCamController.theta = 0.3f;
  const std::vector<std::reference_wrapper<ws::Texture>> texRefs{offscreenFbo.getFirstColorAttachment(), assetManager.white};
  ws::TextureViewer textureViewer{texRefs};
  ws::EditorWindow editorWindow{scene};
  ws::HierarchyWindow hierarchyWindow{scene};
  ws::InspectorWindow inspectorWindow{};
  workshop.shadersToReload = {assetManager.shaders.at("phong"), assetManager.shaders.at("solid"), assetManager.shaders.at("boilerplate")};
  
   
  while (!workshop.shouldStop()) {
    workshop.beginFrame();
    const glm::uvec2 winSize = workshop.getWindowSize();
    offscreenFbo.resizeIfNeeded(winSize.x, winSize.y); // can be resized by something else

    ImGui::Begin("Bloom");
    static glm::vec3 bgColor{42 / 256.0, 96 / 256.0, 87 / 256.0};
    static bool debugScene = false;
    ImGui::Checkbox("Debug Scene using debug shader", &debugScene);
    ImGui::ColorEdit3("BG Color", glm::value_ptr(bgColor));
    ImGui::Separator();
    ImGui::End();

    orbitingCamController.update(workshop.getFrameDurationMs() * 0.001f);
    cam.aspectRatio = static_cast<float>(winSize.x) / winSize.y;

    offscreenFbo.bind();
    glViewport(0, 0, winSize.x, winSize.y);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    glClearColor(bgColor.x, bgColor.y, bgColor.z, 1);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    for (auto& renderable : scene.renderables) {
      ws::Shader& shader = debugScene ? debugShader : renderable.get().shader;
      shader.bind();
      shader.setMatrix4("u_ViewFromWorld", cam.getViewFromWorld());
      shader.setMatrix4("u_ProjectionFromView", cam.getProjectionFromView());
      shader.setVector3("u_CameraPosition", cam.position);
      if (debugScene)
        shader.setVector2("u_CameraNearFar", glm::vec2{cam.nearClip, cam.farClip});
	    renderable.get().texture.bindToUnit(0);
	    renderable.get().texture2.bindToUnit(1);
      shader.setMatrix4("u_WorldFromObject", renderable.get().transform.getWorldFromObjectMatrix());
      renderable.get().mesh.bind();
      renderable.get().mesh.draw();
      renderable.get().mesh.unbind();
	    renderable.get().texture.unbindFromUnit(0);
	    renderable.get().texture2.unbindFromUnit(1);
      shader.unbind();
    }
    offscreenFbo.unbind();

    glViewport(0, 0, winSize.x, winSize.y);
    glClearColor(0, 0, 0, 0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    assetManager.shaders.at("copy").bind();
    offscreenFbo.getFirstColorAttachment().bindToUnit(0);
    assetManager.drawWithEmptyVao(6);
    assetManager.shaders.at("copy").unbind();    

 	  workshop.drawUI();
    textureViewer.draw();
    ws::VObjectPtr clickedObject = editorWindow.draw();
    ws::VObjectPtr selectedObject = hierarchyWindow.draw(clickedObject);
    inspectorWindow.inspectObject(selectedObject);

    workshop.endFrame();
  }

  std::println("Bye!");
  return 0;
}