#include <Workshop/Assets.hpp>
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

class AssetManager {
 public:
  std::unordered_map<std::string, ws::Mesh> meshes;
  std::unordered_map<std::string, ws::Texture> textures;
  std::unordered_map<std::string, ws::Shader> shaders;
};

int main() {
  std::println("Hi!");
  ws::Workshop workshop{1920, 1080, "Infinite Grid"};

  AssetManager assetManager;
  assetManager.meshes.emplace("monkey", ws::loadOBJ(ws::ASSETS_FOLDER / "models/suzanne.obj"));
  assetManager.meshes.emplace("cube", ws::loadOBJ(ws::ASSETS_FOLDER / "models/cube.obj"));
  assetManager.textures.emplace("uv_grid", ws::ASSETS_FOLDER / "images/Wikipedia/UV_checker_Map_byValle.jpg");
  assetManager.textures.emplace("checkerboard", ws::ASSETS_FOLDER / "images/Wikipedia/checkerboard_pattern.png");
  assetManager.textures.emplace("wood", ws::ASSETS_FOLDER / "images/LearnOpenGL/container.jpg");
  ws::Texture whiteTex{ws::Texture::Specs{1, 1, ws::Texture::Format::RGB8, ws::Texture::Filter::Linear}};
  std::vector<uint32_t> whiteTexPixels = {0xFFFFFF};
  whiteTex.uploadPixels(whiteTexPixels.data());
  assetManager.shaders.emplace("phong", ws::Shader{ws::ASSETS_FOLDER / "shaders/phong.vert", ws::ASSETS_FOLDER / "shaders/phong.frag"});
  assetManager.shaders.emplace("unlit", ws::Shader{ws::ASSETS_FOLDER / "shaders/unlit.vert", ws::ASSETS_FOLDER / "shaders/unlit.frag"});
  assetManager.shaders.emplace("grid", ws::Shader{SRC / "grid.vert", SRC / "grid.frag"});
  ws::Shader debugShader{ws::ASSETS_FOLDER / "shaders/debug.vert", ws::ASSETS_FOLDER / "shaders/debug.frag"};
  ws::Framebuffer offscreenFbo = ws::Framebuffer::makeDefaultColorOnly(1, 1);

  ws::RenderableObject monkey = {
      {"Monkey", {glm::vec3{0, -.15f, 0}, glm::vec3{1, 0, 0}, glm::radians(-30.f), glm::vec3{1.5f, 1.5f, 1.5f}}},
      assetManager.meshes.at("monkey"),
      assetManager.shaders.at("phong"),
      assetManager.textures.at("uv_grid"),
      whiteTex,
  };
  ws::RenderableObject box = {
      {"Box", {glm::vec3{1.6f, 0, 2.2f}, glm::vec3{0, 1, 0}, glm::radians(-22.f), glm::vec3{1.f, 2.f, 2.f}}},
      assetManager.meshes.at("cube"),
      assetManager.shaders.at("unlit"),
      assetManager.textures.at("wood"),
      whiteTex,
  };
  ws::PerspectiveCamera3D cam;
  ws::Scene scene{
    .renderables{monkey, box},
  };
  ws::setParent(&monkey, &scene.root);
  ws::setParent(&box, &scene.root);

  uint32_t gridVao;
  glGenVertexArrays(1, &gridVao);

  //ws::AutoOrbitingCamera3DViewController orbitingCamController{cam};
  //orbitingCamController.radius = 10.f;
  //orbitingCamController.theta = 0.3f;
  ws::ManualCamera3DViewController manualCamController{cam};
  const std::vector<std::reference_wrapper<ws::Texture>> texRefs{offscreenFbo.getFirstColorAttachment()};
  ws::TextureViewer textureViewer{texRefs};
  ws::HierarchyWindow hierarchyWindow{scene};
  ws::InspectorWindow inspectorWindow{};
  workshop.shadersToReload = {assetManager.shaders.at("phong"), assetManager.shaders.at("unlit"), assetManager.shaders.at("grid"), debugShader};
  
  glEnable(GL_DEPTH_TEST);
  
  while (!workshop.shouldStop()) {
    workshop.beginFrame();
    const glm::uvec2 winSize = workshop.getWindowSize();
    offscreenFbo.resizeIfNeeded(winSize.x, winSize.y); // can be resized by something else

    ImGui::Begin("Infinite Grid");
    static glm::vec3 bgColor{42 / 256.0, 96 / 256.0, 87 / 256.0};
    static bool debugScene = false;
    ImGui::Checkbox("Debug Scene using debug shader", &debugScene);
    ImGui::ColorEdit3("BG Color", glm::value_ptr(bgColor));
    ImGui::Separator();
    ImGui::End();

    // orbitingCamController.update(workshop.getFrameDurationMs() * 0.001f);
    glm::vec2 cursorPos;
    {
      double xpos, ypos;
      glfwGetCursorPos(workshop.getGLFWwindow(), &xpos, &ypos);
      cursorPos = {xpos, ypos};
    }
    
    manualCamController.update(cursorPos, workshop.mouseState);
    cam.aspectRatio = static_cast<float>(winSize.x) / winSize.y;

    glViewport(0, 0, winSize.x, winSize.y);
    //glEnable(GL_CULL_FACE);
    //glCullFace(GL_BACK);
    glDisable(GL_CULL_FACE);
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    glClearColor(bgColor.x, bgColor.y, bgColor.z, 1);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    {
      ws::Shader& shader = assetManager.shaders.at("grid");
      shader.bind();
      shader.setMatrix4("u_ViewFromWorld", cam.getViewFromWorld());
      shader.setMatrix4("u_ProjectionFromView", cam.getProjectionFromView());
      //shader.setVector3("u_CameraPosition", cam.getPosition());
      glBindVertexArray(gridVao);
      glDrawArrays(GL_TRIANGLES, 0, 6);
      glBindVertexArray(0);
      shader.unbind();
    }
    for (auto& renderable : scene.renderables) {
      ws::Shader& shader = debugScene ? debugShader : renderable.get().shader;
      shader.bind();
      shader.setMatrix4("u_ViewFromWorld", cam.getViewFromWorld());
      shader.setMatrix4("u_ProjectionFromView", cam.getProjectionFromView());
      shader.setVector3("u_CameraPosition", cam.getPosition());
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

 	  workshop.drawUI();
    textureViewer.draw();
    ws::VObjectPtr selectedObject = hierarchyWindow.draw();
    inspectorWindow.inspectObject(selectedObject);

    workshop.endFrame();
  }

  std::println("Bye!");
  return 0;
}