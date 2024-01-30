#include <Workshop/Assets.hpp>
#include <Workshop/Camera.hpp>
#include <Workshop/Framebuffer.hpp>
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

class AssetManager {
 public:
  std::unordered_map<std::string, ws::Mesh> meshes;
  std::unordered_map<std::string, ws::Texture> textures;
  std::unordered_map<std::string, ws::Shader> shaders;
};

struct AmbientLight {
  glm::vec3 color;
  float _pad0;
};

struct HemisphericalLight {
  glm::vec3 northColor;
  float intensity;
  glm::vec3 southColor;
  float _pad0;
};

const int MAX_POINT_LIGHTS = 8;
struct PointLight {
  glm::vec3 position;
  float intensity;
  glm::vec3 color;
  float _pad0;
};

const int MAX_DIRECTIONAL_LIGHTS = 4;
struct DirectionalLight {
  glm::vec3 position;
  float intensity;
  //
  glm::vec3 direction;
  float _pad0;
  //
  glm::vec3 color;
  float _pad1;
};


struct SceneUniforms {
  glm::mat4 u_ProjectionFromView;
  glm::mat4 u_ViewFromWorld;
  //
  glm::vec3 u_CameraPosition;
  float _pad0;
  //
  AmbientLight ambientLight;
  //
  HemisphericalLight hemisphericalLight;
  //
  glm::vec3 _pad1;
  int numPointLights;
  //
  PointLight pointLights[MAX_POINT_LIGHTS];
  //
  glm::vec3 _pad2;
  int numDirectionalLights;
  //
  DirectionalLight directionalLights[MAX_DIRECTIONAL_LIGHTS];
};

int main() {
  std::println("Hi!");
  ws::Workshop workshop{1920, 1080, "Material System"};

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
  assetManager.shaders.emplace("boilerplate", ws::Shader{SRC / "boilerplate.vert", SRC / "boilerplate.frag"});
  ws::Shader debugShader{ws::ASSETS_FOLDER / "shaders/debug.vert", ws::ASSETS_FOLDER / "shaders/debug.frag"};
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
      assetManager.shaders.at("phong"),
      assetManager.textures.at("checkerboard"),
      whiteTex,
  };
  ws::RenderableObject box = {
      {"Box", {glm::vec3{1.6f, 0, 2.2f}, glm::vec3{0, 1, 0}, glm::radians(-22.f), glm::vec3{1.f, 2.f, 2.f}}},
      assetManager.meshes.at("cube"),
      assetManager.shaders.at("boilerplate"),
      assetManager.textures.at("wood"),
      assetManager.textures.at("checkerboard"),
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
  const std::vector<std::reference_wrapper<ws::Texture>> texRefs{offscreenFbo.getFirstColorAttachment()};
  ws::TextureViewer textureViewer{texRefs};
  ws::EditorWindow editorWindow{scene};
  ws::HierarchyWindow hierarchyWindow{scene};
  ws::InspectorWindow inspectorWindow{};
  workshop.shadersToReload = {assetManager.shaders.at("phong"), assetManager.shaders.at("unlit"), assetManager.shaders.at("boilerplate"), debugShader};
  
  glEnable(GL_DEPTH_TEST);
  ws::UniformBuffer<SceneUniforms> sceneUbo{1};
  sceneUbo.compareSizeWithUniformBlock(assetManager.shaders.at("boilerplate").getId(), "SceneUniforms");
  
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
    ImGui::End();

    orbitingCamController.update(workshop.getFrameDurationMs() * 0.001f);
    cam.aspectRatio = static_cast<float>(winSize.x) / winSize.y;

    sceneUbo.uniforms.u_ViewFromWorld = cam.getViewFromWorld();
    sceneUbo.uniforms.u_ProjectionFromView = cam.getProjectionFromView();
    sceneUbo.uniforms.u_CameraPosition = cam.position;
    sceneUbo.uniforms.ambientLight.color = glm::vec3(0.05, 0.0, 0.05);
    sceneUbo.uniforms.hemisphericalLight.northColor = glm::vec3(0.05, 0.15, 0.95);
    sceneUbo.uniforms.hemisphericalLight.southColor = glm::vec3(0.85, 0.75, 0.05);
    sceneUbo.uniforms.hemisphericalLight.intensity = 1.0f;
    sceneUbo.uniforms.numPointLights = 1;
    sceneUbo.uniforms.pointLights[0].position = glm::vec3(0, 0, 3);
    sceneUbo.uniforms.pointLights[0].color = glm::vec3(1, 1, 1);
    sceneUbo.uniforms.pointLights[0].intensity = 1.f;
    sceneUbo.uniforms.numDirectionalLights = 1;
    sceneUbo.uniforms.directionalLights[0].position = glm::vec3(1, 1, 1);
    sceneUbo.uniforms.directionalLights[0].intensity = 0.5f;
    sceneUbo.uniforms.directionalLights[0].direction = glm::vec3(-1, -1, -1);
    sceneUbo.uniforms.directionalLights[0].color = glm::vec3(1, 1, 1);
    sceneUbo.upload();

    offscreenFbo.bind();
    glViewport(0, 0, winSize.x, winSize.y);
    glDisable(GL_CULL_FACE);
    glClearColor(0, 0, 0, 0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    assetManager.shaders.at("boilerplate").bind();
    for (auto& renderable : scene.renderables) {
      renderable.get().texture.bindToUnit(0);
      assetManager.shaders.at("boilerplate").setMatrix4("u_WorldFromObject", renderable.get().transform.getWorldFromObjectMatrix());
      const ws::Mesh& mesh = renderable.get().mesh;
      mesh.bind();
      mesh.draw();
      mesh.unbind();
      renderable.get().texture.unbindFromUnit(0);
    }
    assetManager.shaders.at("boilerplate").unbind();
    offscreenFbo.unbind();

    glViewport(0, 0, winSize.x, winSize.y);
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