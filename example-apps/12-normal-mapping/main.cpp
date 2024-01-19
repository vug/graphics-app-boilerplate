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
  ws::Workshop workshop{1920, 1080, "Boilerplate app"};

  AssetManager assetManager;
  assetManager.meshes.emplace("monkey", ws::loadOBJ(ws::ASSETS_FOLDER / "models/suzanne.obj"));
  assetManager.meshes.emplace("cube", ws::loadOBJ(ws::ASSETS_FOLDER / "models/cube.obj"));
  // https://cgaxis.com/product/old-brick-wall-pbr-texture-31/
  assetManager.textures.emplace("brickwall", ws::ASSETS_FOLDER / "images/cgaxis/cgaxis_pbr_17_old_brick_wall_22_diffuse.jpg");
  assetManager.textures.emplace("brickwall_normal", ws::ASSETS_FOLDER / "images/cgaxis/cgaxis_pbr_17_old_brick_wall_22_normal.jpg");
  // https://cgaxis.com/product/hedgehog-fur-pbr-texture/
  assetManager.textures.emplace("hedgehog", ws::ASSETS_FOLDER / "images/cgaxis/hedgehog_fur_37_34_diffuse.jpg");
  assetManager.textures.emplace("hedgehog_normal", ws::ASSETS_FOLDER / "images/cgaxis/hedgehog_fur_37_34_normal.jpg");
  // https://cgaxis.com/product/old-basement-floor-6127/
  assetManager.textures.emplace("wood_floor", ws::ASSETS_FOLDER / "images/cgaxis/old_basement_floor_61_27_basecolor_diffuse.png");
  assetManager.textures.emplace("wood_floor_normal", ws::ASSETS_FOLDER / "images/cgaxis/old_basement_floor_61_27_normal_opengl.png");
  ws::Texture whiteTex{ws::Texture::Specs{1, 1, ws::Texture::Format::RGB8, ws::Texture::Filter::Linear}};
  std::vector<uint32_t> whiteTexPixels = {0xFFFFFF};
  whiteTex.uploadPixels(whiteTexPixels.data());
  assetManager.shaders.emplace("phong", ws::Shader{ws::ASSETS_FOLDER / "shaders/phong.vert", ws::ASSETS_FOLDER / "shaders/phong.frag"});
  assetManager.shaders.emplace("unlit", ws::Shader{ws::ASSETS_FOLDER / "shaders/unlit.vert", ws::ASSETS_FOLDER / "shaders/unlit.frag"});
  assetManager.shaders.emplace("normal_mapped", ws::Shader{SRC / "normal_mapped.vert", SRC / "normal_mapped.frag"});

  ws::RenderableObject ground = {
      {"Ground", {glm::vec3{0, -1, 0}, glm::vec3{0, 0, 1}, 0, glm::vec3{20.f, .1f, 20.f}}},
      assetManager.meshes.at("cube"),
      assetManager.shaders.at("normal_mapped"),
      assetManager.textures.at("wood_floor"),
      assetManager.textures.at("wood_floor_normal"),
  };
  ws::RenderableObject monkey = {
      {"Monkey", {glm::vec3{0, -.15f, 0}, glm::vec3{1, 0, 0}, glm::radians(-30.f), glm::vec3{1.5f, 1.5f, 1.5f}}},
      assetManager.meshes.at("monkey"),
      assetManager.shaders.at("normal_mapped"),
      assetManager.textures.at("hedgehog"),
      assetManager.textures.at("hedgehog_normal"),
  };
  ws::RenderableObject box = {
      {"Box", {glm::vec3{1.6f, 0, 2.2f}, glm::vec3{0, 1, 0}, glm::radians(-22.f), glm::vec3{1.f, 2.f, 2.f}}},
      assetManager.meshes.at("cube"),
      assetManager.shaders.at("normal_mapped"),
      assetManager.textures.at("brickwall"),
      assetManager.textures.at("brickwall_normal"),
  };
  ws::Camera cam;
  ws::Scene scene{
      .renderables{ground, monkey, box},
  };
  ws::setParent(&ground, &scene.root);
  ws::setParent(&monkey, &scene.root);
  ws::setParent(&box, &scene.root);

  ws::AutoOrbitingCameraController orbitingCamController{cam};
  orbitingCamController.radius = 10.f;
  orbitingCamController.theta = 0.3f;
  orbitingCamController.speed = 0.f;
  const std::vector<std::reference_wrapper<ws::Texture>> texRefs{};
  ws::EditorWindow editorWindow{scene};
  ws::HierarchyWindow hierarchyWindow{scene};
  ws::InspectorWindow inspectorWindow{};
  workshop.shadersToReload = {assetManager.shaders.at("phong"), assetManager.shaders.at("unlit"), assetManager.shaders.at("normal_mapped")};
  
  glEnable(GL_DEPTH_TEST);
  
  while (!workshop.shouldStop()) {
    workshop.beginFrame();
    const glm::uvec2 winSize = workshop.getWindowSize();

    ImGui::Begin("Normal Mapping");
    static glm::vec3 bgColor{42 / 256.0, 96 / 256.0, 87 / 256.0};
    ImGui::ColorEdit3("BG Color", glm::value_ptr(bgColor));
    ImGui::Separator();

    static float normalMapAmount = 1.f;
    ImGui::SliderFloat("Normal Map Amount", &normalMapAmount, 0, 1);
    static int shadingMode = 0;
    std::array<const char*, 7> items = {"Scene", "Diffuse Map", "Normal Map", "Vertex Normal (World)", "Map Normal (World)", "Tangent", "Bi-Tangent"};
    ImGui::Combo("Shading Mode", &shadingMode, items.data(), static_cast<int>(items.size()));
    static bool hasSpecular = true;
    ImGui::Checkbox("Has Specular", &hasSpecular);
    static bool shouldUseWhiteAsSpecular = false;
    ImGui::Checkbox("Ignore Diffuse Map, just use white", &shouldUseWhiteAsSpecular);
    static bool shouldIgnoreNormalMap = false;
    ImGui::Checkbox("Ignore Normal Map, just use vertex normal", &shouldIgnoreNormalMap);

    ImGui::End();

    orbitingCamController.update(workshop.getFrameDurationMs() * 0.001f);
    cam.aspectRatio = static_cast<float>(winSize.x) / winSize.y;

    glViewport(0, 0, winSize.x, winSize.y);
    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    glClearColor(bgColor.x, bgColor.y, bgColor.z, 1);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    for (auto& renderable : scene.renderables) {
      ws::Shader& shader = renderable.get().shader;
      shader.bind();
      shader.setFloat("u_AmountOfMapNormal", normalMapAmount);
      shader.setInteger("u_ShadingMode", shadingMode);
      shader.setInteger("u_HasSpecular", hasSpecular);
      shader.setInteger("u_UseWhiteAsDiffuse", shouldUseWhiteAsSpecular);
      shader.setInteger("u_IgnoreNormalMap", shouldIgnoreNormalMap);
      shader.setMatrix4("u_ViewFromWorld", cam.getViewFromWorld());
      shader.setMatrix4("u_ProjectionFromView", cam.getProjectionFromView());
      shader.setVector3("u_CameraPosition", cam.position);
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
    ws::VObjectPtr clickedObject = editorWindow.draw();
    ws::VObjectPtr selectedObject = hierarchyWindow.draw(clickedObject);
    inspectorWindow.inspectObject(selectedObject);

    workshop.endFrame();
  }

  std::println("Bye!");
  return 0;
}