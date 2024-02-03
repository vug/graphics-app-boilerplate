#include <Workshop/AssetManager.hpp>
#include <Workshop/Assets.hpp>
#include <Workshop/Camera.hpp>
#include <Workshop/Framebuffer.hpp>
#include <Workshop/Lights.hpp>
#include <Workshop/Material.hpp>
#include <Workshop/Model.hpp>
#include <Workshop/Scene.hpp>
#include <Workshop/Shader.hpp>
#include <Workshop/UI.hpp>
#include <Workshop/UniformBuffer.hpp>
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
  ws::Workshop workshop{1920, 1080, "Material System"};

  ws::AssetManager assetManager;
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
  assetManager.shaders.emplace("checkered", ws::Shader{SRC / "boilerplate.vert", SRC / "boilerplate.frag"});
  assetManager.materials.emplace("checkered1", 
    ws::Material{
      .shader = assetManager.shaders.at("checkered"),
      .parameters = {
          {"color1", glm::vec3(1, 0, 0)},
          {"color2", glm::vec3(0, 0, 1)},
          {"numCells", 2},
          {"mainTex", assetManager.textures.at("uv_grid")},
      }
    }
  );
  assetManager.materials.emplace("checkered2",
    ws::Material{
      .shader = assetManager.shaders.at("checkered"),
      .parameters = {
          {"color1", glm::vec3(1, 1, 0)},
          {"color2", glm::vec3(0, 1, 1)},
          {"numCells", 3},
          {"mainTex", assetManager.textures.at("wood")},
      }
    }
  );
  ws::Framebuffer offscreenFbo;

  ws::RenderableObject ground = {
      {"Ground", {glm::vec3{0, -1, 0}, glm::vec3{0, 0, 1}, 0, glm::vec3{20.f, .1f, 20.f}}},
      assetManager.meshes.at("cube"),
      assetManager.shaders.at("phong"),
      assetManager.textures.at("uv_grid"),
      whiteTex,
  };
  ws::RenderableObject monkey = {
      {"Monkey", {glm::vec3{0, -.15f, 0}, glm::vec3{1, 0, 0}, glm::radians(-30.f), glm::vec3{1.5f, 1.5f, 1.5f}}},
      assetManager.meshes.at("monkey"),
      assetManager.shaders.at("checkered"),
      whiteTex,
      whiteTex,
  };
  ws::RenderableObject box = {
      {"Box", {glm::vec3{1.6f, 0, 2.2f}, glm::vec3{0, 1, 0}, glm::radians(-22.f), glm::vec3{1.f, 2.f, 2.f}}},
      assetManager.meshes.at("cube"),
      assetManager.shaders.at("checkered"),
      whiteTex,
      whiteTex,
  };

  ws::Scene scene{
    .renderables{ground, monkey, box},
    .ambientLight = ws::AmbientLight{.color = glm::vec3(0.05, 0.0, 0.05)},
    .hemisphericalLight = ws::HemisphericalLight{
      .northColor = glm::vec3(0.05, 0.15, 0.95),
      .intensity = 1.0f,
      .southColor = glm::vec3(0.85, 0.75, 0.05),
    },
    .pointLights = std::vector<ws::PointLight>{
      ws::PointLight{
        .position = glm::vec3(0, 0, 3),
        .intensity = 1.f,
        .color = glm::vec3(1, 1, 1),
      },
    },
    .directionalLights = std::vector<ws::DirectionalLight>{
      ws::DirectionalLight{
        .position = glm::vec3(1, 1, 1),
        .intensity = 0.5f,
        .direction = glm::vec3(-1, -1, -1),
        .color = glm::vec3(1, 1, 1),
      },
    },
  };
  ws::setParent(&ground, &scene.root);
  ws::setParent(&monkey, &scene.root);
  ws::setParent(&box, &scene.root);

  ws::AutoOrbitingCameraController orbitingCamController{scene.camera};
  orbitingCamController.radius = 10.f;
  orbitingCamController.theta = 0.3f;
  const std::vector<std::reference_wrapper<ws::Texture>> texRefs{offscreenFbo.getFirstColorAttachment()};
  ws::TextureViewer textureViewer{texRefs};
  ws::EditorWindow editorWindow{scene};
  ws::HierarchyWindow hierarchyWindow{scene};
  ws::InspectorWindow inspectorWindow{};
  workshop.shadersToReload = {assetManager.shaders.at("phong"), assetManager.shaders.at("unlit"), assetManager.shaders.at("checkered")};
  
  glEnable(GL_DEPTH_TEST);
  scene.ubo.compareSizeWithUniformBlock(assetManager.shaders.at("checkered").getId(), "SceneUniforms");
 
  while (!workshop.shouldStop()) {
    workshop.beginFrame();
    const glm::uvec2 winSize = workshop.getWindowSize();
    offscreenFbo.resizeIfNeeded(winSize.x, winSize.y); // can be resized by something else

    ImGui::Begin("Material System");
    static glm::vec3 bgColor{42 / 256.0, 96 / 256.0, 87 / 256.0};
    static bool debugScene = false;
    ImGui::Checkbox("Debug Scene using debug shader", &debugScene);
    ImGui::ColorEdit3("BG Color", glm::value_ptr(bgColor));
    ImGui::Separator();
    // bring vector of references to texture to material widget (?)
    ImGui::Text("Material1");
    ImGuiMaterialWidget(assetManager.materials.at("checkered1"), assetManager);
    if (ImGui::Button("Param/Unif Match1"))
      assetManager.materials.at("checkered1").doParametersAndUniformsMatch();
    ImGui::SameLine();
    if (ImGui::Button("Print Params1"))
      std::println("{}", assetManager.materials.at("checkered1").parametersToString());
    ImGui::Text("Material2");
    ImGuiMaterialWidget(assetManager.materials.at("checkered2"), assetManager);
    if (ImGui::Button("Param/Unif Math2"))
      assetManager.materials.at("checkered2").doParametersAndUniformsMatch();
    ImGui::SameLine();
    if (ImGui::Button("Print Params2"))
      std::println("{}", assetManager.materials.at("checkered2").parametersToString());
    ImGui::End();

    orbitingCamController.update(workshop.getFrameDurationMs() * 0.001f);
    scene.camera.aspectRatio = static_cast<float>(winSize.x) / winSize.y;
    scene.uploadUniforms();

    offscreenFbo.bind();
    glViewport(0, 0, winSize.x, winSize.y);
    glDisable(GL_CULL_FACE);
    glClearColor(0, 0, 0, 0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    assetManager.shaders.at("checkered").bind();
    for (auto& renderable : scene.renderables) {
      if (renderable.get().name == "Monkey")
        assetManager.materials.at("checkered1").uploadParameters();
      else if (renderable.get().name == "Box")
        assetManager.materials.at("checkered2").uploadParameters();
      else {
        renderable.get().texture.bindToUnit(3);
        renderable.get().texture2.bindToUnit(7);
      }
      assetManager.shaders.at("checkered").setMatrix4("u_WorldFromObject", renderable.get().transform.getWorldFromObjectMatrix());
      const ws::Mesh& mesh = renderable.get().mesh;
      mesh.bind();
      mesh.draw();
      mesh.unbind();
      renderable.get().texture.unbindFromUnit(0);
    }
    assetManager.shaders.at("checkered").unbind();
    offscreenFbo.unbind();

    glViewport(0, 0, winSize.x, winSize.y);
    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    glClearColor(bgColor.x, bgColor.y, bgColor.z, 1);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    for (auto& renderable : scene.renderables) {
      ws::Shader& shader = renderable.get().shader;
      shader.bind();
      if (renderable.get().name == "Monkey")
        assetManager.materials.at("checkered1").uploadParameters();
      else if (renderable.get().name == "Box")
        assetManager.materials.at("checkered2").uploadParameters();
      else {
        renderable.get().texture.bindToUnit(3);
        renderable.get().texture2.bindToUnit(7);
      }
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
    static ws::VObjectPtr selectedObject;
    ws::VObjectPtr clickedObject = editorWindow.draw(selectedObject, workshop.getFrameDurationSec());
    selectedObject = hierarchyWindow.draw(clickedObject);
    inspectorWindow.inspectObject(selectedObject);

    workshop.endFrame();
  }

  std::println("Bye!");
  return 0;
}