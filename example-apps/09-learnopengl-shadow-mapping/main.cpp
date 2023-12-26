#include <Workshop/Assets.hpp>
#include <Workshop/Camera.hpp>
#include <Workshop/Framebuffer.hpp>
#include <Workshop/Model.hpp>
#include <Workshop/Scene.hpp>
#include <Workshop/Shader.hpp>
#include <Workshop/Texture.hpp>
#include <Workshop/Transform.hpp>
#include <Workshop/UI.hpp>
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

const std::filesystem::path SRC{SOURCE_DIR};

class AssetManager {
 public:
  std::unordered_map<std::string, ws::Mesh> meshes;
  std::unordered_map<std::string, ws::Texture> textures;
  std::unordered_map<std::string, ws::Framebuffer> framebuffers;
  std::unordered_map<std::string, ws::Shader> shaders;
};

struct DirectionalLight {
  glm::vec3 position{};
  glm::vec3 target{};
  float intensity = 1.0f;
  // Shadow Parameters
  uint32_t shadowWidth = 1024;
  uint32_t shadowHeight = 1024;
  glm::vec2 shadowBias = {0.005f, 0.05f};
  float side = 10.f;
  float near = 1.0f;
  float far = 7.5f;

  glm::mat4 getProjection() const {
    return glm::ortho(-side, side, -side, side, near, far);
  }

  glm::mat4 getView() const {
    return glm::lookAt(position, target, glm::vec3{0, 1, 0});
  }

  glm::mat4 getLightSpaceMatrix() const {
    return getProjection() * getView();
  }
};


int main() {
  std::println("Hi!");
  ws::Workshop workshop{2048, 1536, "Shadow Maps"};

  AssetManager assetManager;
  assetManager.meshes.emplace("cube", ws::Mesh{ws::loadOBJ(ws::ASSETS_FOLDER / "models/cube.obj")});
  assetManager.meshes.emplace("quad", ws::Mesh{ws::loadOBJ(ws::ASSETS_FOLDER / "models/quad.obj")});
  assetManager.textures.emplace("wood", ws::Texture{ws::ASSETS_FOLDER / "images/LearnOpenGL/container.jpg"});
  assetManager.shaders.emplace("unlit", ws::Shader{ws::ASSETS_FOLDER / "shaders/unlit.vert", ws::ASSETS_FOLDER / "shaders/unlit.frag"});
  assetManager.shaders.emplace("simpleDepth", ws::Shader{SRC / "shadow_mapping_depth.vert", SRC / "shadow_mapping_depth.frag"});
  assetManager.shaders.emplace("phongShadowed", ws::Shader{SRC / "phong_shadowed.vert", SRC / "phong_shadowed.frag"});
  assetManager.shaders.emplace("depthViz", ws::Shader{ws::ASSETS_FOLDER / "shaders/fullscreen_quad_without_vbo.vert", SRC / "depth_viz.frag"});
  // TODO: weirdly I need a move, can't emplace an FB directly
  ws::Framebuffer fbo{1, 1, false};
  assetManager.framebuffers.emplace("shadowFBO", std::move(fbo));
  uint32_t dummyVao;
  glGenVertexArrays(1, &dummyVao);

  ws::RenderableObject ground = {
    ws::Object{std::string{"Ground"}, ws::Transform{glm::vec3{0, -0.5, 0}, glm::vec3{0, 1, 0}, 0, glm::vec3{25.f, 1, 25.f}}},
    assetManager.meshes.at("quad"),
    assetManager.shaders["phongShadowed"],
    assetManager.textures["wood"],
  };
  ws::RenderableObject cube1 = {
    {"Cube1", {glm::vec3{0, 1.5f, 0}, glm::vec3{0, 0, 1}, 0, glm::vec3{1.f, 1.f, 1.f}}},
    assetManager.meshes.at("cube"),
    assetManager.shaders["phongShadowed"],
    assetManager.textures["wood"],
  };
  ws::RenderableObject cube2 = {
    ws::Object{std::string{"Cube2"}, ws::Transform{glm::vec3{2.0f, 0.0f, 1.0f}, glm::vec3{0, 0, 1}, 0, glm::vec3{1.f, 1.f, 1.f}}},
    assetManager.meshes.at("cube"),
    assetManager.shaders["phongShadowed"],
    assetManager.textures["wood"],
  };
  ws::RenderableObject cube3 = {
    ws::Object{std::string{"Cube3"}, ws::Transform{glm::vec3{-1.f, 0, 2.f}, glm::normalize(glm::vec3{1.f, 0, 1.f}), glm::radians(60.f), glm::vec3{.5f, .5f, .5f}}},
    assetManager.meshes.at("cube"),
    assetManager.shaders["phongShadowed"],
    assetManager.textures["wood"],
  };
  ws::CameraObject cam1{
    ws::Object{std::string{"SceneCamera"}, ws::Transform{glm::vec3{0, 0, -5.f}, glm::vec3{0, 0, 1}, 0, glm::vec3{1, 1, 1}}},
  };
  ws::Scene scene{
    .renderables{ground, cube1, cube2, cube3},
    .cameras{cam1}
  };
  ws::setParent(&ground, &scene.root);
  ws::setParent(&cube1, &scene.root);
  ws::setParent(&cube2, &scene.root);
  ws::setParent(&cube3, &scene.root);
  ws::setParent(&cam1, &scene.root);

  ws::PerspectiveCamera3D& cam = scene.cameras[0].get().camera;
  ws::AutoOrbitingCamera3DViewController orbitingCamController{cam};
  orbitingCamController.radius = 7.7f;
  orbitingCamController.theta = 0.5;

  DirectionalLight light;
  light.position = {-2.f, 4.f, -1.f};
  assetManager.framebuffers.at("shadowFBO").resizeIfNeeded(light.shadowWidth, light.shadowHeight);

  const std::vector<std::reference_wrapper<ws::Texture>> texRefs{assetManager.framebuffers.at("shadowFBO").getDepthAttachment()};
  ws::TextureViewer textureViewer{texRefs};
  ws::HierarchyWindow hierarchyWindow{scene};
  ws::InspectorWindow inspectorWindow{};

  glEnable(GL_DEPTH_TEST);

  while (!workshop.shouldStop()) {
    workshop.beginFrame();
    const glm::uvec2 winSize = workshop.getWindowSize();

    workshop.imGuiDrawAppWindow();

    ImGui::Begin("Main");
    static glm::vec3 bgColor{144.f/255, 225.f/255, 236.f/255};
    ImGui::ColorEdit3("BG Color", glm::value_ptr(bgColor));
    ImGui::Separator();
    ImGui::Text("Light");
    ImGui::DragFloat3("Position", glm::value_ptr(light.position));
    ImGui::DragFloat3("Target", glm::value_ptr(light.target));
    ImGui::DragFloat("Intensity", &light.intensity);
    ImGui::DragFloat("Shadow Frustrum Side", &light.side);
    uint32_t minDim = 16;
    uint32_t maxDim = 4096;
    ImGui::DragScalar("Shadow Map Width", ImGuiDataType_U32, &light.shadowWidth, 1.0f, &minDim, &maxDim);
    ImGui::DragScalar("Shadow Map Height", ImGuiDataType_U32, &light.shadowHeight, 1.0f, &minDim, &maxDim);
    ImGui::DragFloat2("Shadow Bias", glm::value_ptr(light.shadowBias));
    ImGui::Separator();
    ImGui::End();

    assetManager.framebuffers.at("shadowFBO").resizeIfNeeded(light.shadowWidth, light.shadowHeight);
    orbitingCamController.update(0.01f);
    cam.aspectRatio = static_cast<float>(winSize.x) / winSize.y;

    glClearColor(bgColor.x, bgColor.y, bgColor.z, 1);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    auto drawScene = [&]() {
      glViewport(0, 0, winSize.x, winSize.y);
      for (auto& renderable : scene.renderables) {
        ws::Shader& shader = renderable.get().shader;
        shader.bind();
        renderable.get().mesh.bind();
        glBindTextureUnit(0, renderable.get().texture.getId());
        glBindTextureUnit(1, assetManager.framebuffers.at("shadowFBO").getDepthAttachment().getId());
        shader.setVector3("u_CameraPos", cam.getPosition());
        shader.setMatrix4("u_ViewFromWorld", cam.getViewFromWorld());
        shader.setMatrix4("u_ProjectionFromView", cam.getProjectionFromView());
        shader.setMatrix4("u_LightSpaceMatrix", light.getLightSpaceMatrix());
        shader.setVector3("u_LightPos", light.position);
        shader.setFloat("u_LightIntensity", light.intensity);
        shader.setVector2("u_ShadowBias", light.shadowBias);
        shader.setMatrix4("u_WorldFromObject", renderable.get().transform.getWorldFromObjectMatrix());
        // TODO: not there yet. Positions and scale inheritence looks fine, but rotation is broken. Parent's rotation should rotate child's coordinate system.
        //shader.setMatrix4("u_WorldFromObject", renderable.get().getGlobalTransformMatrix());
        renderable.get().mesh.draw();
        glBindTextureUnit(0, 0);
        glBindTextureUnit(1, 0);
        renderable.get().mesh.unbind();
        shader.unbind();
      }
    };

    auto drawShadowMap = [&]() {
      glViewport(0, 0, light.shadowWidth, light.shadowHeight);
      assetManager.framebuffers.at("shadowFBO").bind();
      glClear(GL_DEPTH_BUFFER_BIT);
      assetManager.shaders.at("simpleDepth").bind();
      // cam.getProjectionFromView() * cam.getViewFromWorld() to see from camera's perspective
      assetManager.shaders.at("simpleDepth").setMatrix4("u_LightSpaceMatrix", light.getLightSpaceMatrix());

      for (auto& renderable : scene.renderables) {
        renderable.get().mesh.bind();
        assetManager.shaders.at("simpleDepth").setMatrix4("u_WorldFromObject", renderable.get().transform.getWorldFromObjectMatrix());
        renderable.get().mesh.draw();
        renderable.get().mesh.unbind();
      }

      assetManager.shaders.at("simpleDepth").unbind();
      assetManager.framebuffers.at("shadowFBO").unbind();
    };

    auto visualizeDepth = [&]() {
      glViewport(0, 0, winSize.x, winSize.y);
      const ws::Shader& shader = assetManager.shaders.at("depthViz");
      shader.bind();

      shader.setFloat("near_plane", light.near);
      shader.setFloat("far_plane", light.far);
      glBindTextureUnit(0, assetManager.framebuffers.at("shadowFBO").getDepthAttachment().getId());
      glBindVertexArray(dummyVao);
      glDrawArrays(GL_TRIANGLES, 0, 6);
      glBindVertexArray(0);
      shader.unbind();
    };

    drawShadowMap();
    drawScene();
    //visualizeDepth();

    textureViewer.draw();
    ws::VObjectPtr selectedObject = hierarchyWindow.draw();
    inspectorWindow.inspectObject(selectedObject);

    workshop.endFrame();
  }

  std::println("Bye!");
  return 0;
}