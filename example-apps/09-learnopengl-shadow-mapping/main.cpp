#include <Workshop/AssetManager.hpp>
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

  ws::AssetManager assetManager;
  assetManager.meshes.emplace("cube", ws::Mesh{ws::loadOBJ(ws::ASSETS_FOLDER / "models/cube.obj")});
  assetManager.meshes.emplace("quad", ws::Mesh{ws::loadOBJ(ws::ASSETS_FOLDER / "models/quad.obj")});
  assetManager.meshes.emplace("axes", ws::Mesh{ws::loadOBJ(ws::ASSETS_FOLDER / "models/coordinate_axes.obj")});
  assetManager.textures.emplace("wood", ws::Texture{ws::ASSETS_FOLDER / "images/LearnOpenGL/container.jpg"});
  assetManager.textures.emplace("checkerboard", ws::Texture{ws::ASSETS_FOLDER / "images/Wikipedia/checkerboard_pattern.png"});
  assetManager.shaders.emplace("unlit", ws::Shader{ws::ASSETS_FOLDER / "shaders/unlit.vert", ws::ASSETS_FOLDER / "shaders/unlit.frag"});
  assetManager.shaders.emplace("simpleDepth", ws::Shader{SRC / "shadow_mapping_depth.vert", SRC / "shadow_mapping_depth.frag"});
  assetManager.shaders.emplace("phongShadowed", ws::Shader{SRC / "phong_shadowed.vert", SRC / "phong_shadowed.frag"});
  assetManager.shaders.emplace("depthViz", ws::Shader{ws::ASSETS_FOLDER / "shaders/fullscreen_quad_without_vbo.vert", SRC / "depth_viz.frag"});
  assetManager.materials.emplace("phongShadowed-generic", ws::Material{
    .shader = assetManager.shaders.at("phongShadowed"),
    .parameters = {
      {"diffuseTexture", assetManager.textures.at("wood")},
    },
    .shouldMatchUniforms = false,
  });
  //assert(assetManager.doAllMaterialsHaveMatchingParametersAndUniforms());
  float shadowBorderColor[] = {1.f, 0.f, 0.f, 0.f};
  ws::Framebuffer shadowFbo{1, 1, false};
  glTextureParameteri(shadowFbo.getDepthAttachment().getId(), GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
  glTextureParameteri(shadowFbo.getDepthAttachment().getId(), GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
  glTextureParameterfv(shadowFbo.getDepthAttachment().getId(), GL_TEXTURE_BORDER_COLOR, shadowBorderColor);
  uint32_t dummyVao;
  glGenVertexArrays(1, &dummyVao);

  ws::RenderableObject axes = {
    ws::Object{std::string{"Axes"}, ws::Transform{glm::vec3{0, 0, 0}, glm::vec3{0, 1, 0}, 0, glm::vec3{1, 1, 1}}},
    assetManager.meshes.at("axes"),
    assetManager.materials.at("phongShadowed-generic"),
  };
  ws::RenderableObject ground = {
    //ws::Object{std::string{"Ground"}, ws::Transform{glm::vec3{0, -0.5, 0}, glm::vec3{1, 0, 0}, glm::radians(-90.f), glm::vec3{25.f, 25.f, 1.f}}},
    ws::Object{std::string{"Ground"}, ws::Transform{glm::vec3{0, -0.5, 0}, glm::vec3{0, 1, 0}, 0, glm::vec3{25.f, 1, 25.f}}},
    assetManager.meshes.at("quad"),
    assetManager.materials.at("phongShadowed-generic"),
  };
  ws::RenderableObject cube1 = {
    {"Cube1", {glm::vec3{0, 1.5f, 0}, glm::vec3{0, 0, 1}, 0, glm::vec3{1.f, 1.f, 1.f}}},
    assetManager.meshes.at("cube"),
    assetManager.materials.at("phongShadowed-generic"),
  };
  ws::RenderableObject cube2 = {
    ws::Object{std::string{"Cube2"}, ws::Transform{glm::vec3{2.0f, 0.0f, 1.0f}, glm::vec3{0, 0, 1}, 0, glm::vec3{1.f, 1.f, 1.f}}},
    assetManager.meshes.at("cube"),
    assetManager.materials.at("phongShadowed-generic"),
  };
  ws::RenderableObject cube3 = {
    ws::Object{std::string{"Cube3"}, ws::Transform{glm::vec3{-1.f, 0, 2.f}, glm::normalize(glm::vec3{1.f, 0, 1.f}), glm::radians(60.f), glm::vec3{.5f, .5f, .5f}}},
    assetManager.meshes.at("cube"),
    assetManager.materials.at("phongShadowed-generic"),
  };
  ws::Scene scene{
    .renderables{ground, cube1, cube2, cube3, axes},
  };
  ws::setParent(&ground, &scene.root);
  ws::setParent(&cube1, &scene.root);
  ws::setParent(&cube2, &scene.root);
  ws::setParent(&cube3, &scene.root);
  ws::setParent(&axes, &scene.root);
  scene.camera.position = glm::vec3{0, 0, -5.f};
  scene.camera.target = glm::vec3{0, 0, 0};

  ws::Camera& cam = scene.camera;
  ws::AutoOrbitingCameraController orbitingCamController{cam};
  orbitingCamController.radius = 7.7f;
  orbitingCamController.theta = 0.5;

  DirectionalLight light;
  light.position = {-2.f, 4.f, -1.f};
  light.intensity = 3;
  shadowFbo.resizeIfNeeded(light.shadowWidth, light.shadowHeight);

  const std::vector<std::reference_wrapper<ws::Texture>> texRefs{shadowFbo.getDepthAttachment()};
  ws::TextureViewer textureViewer{texRefs};
  ws::HierarchyWindow hierarchyWindow{scene};
  ws::InspectorWindow inspectorWindow{};

  glEnable(GL_DEPTH_TEST);
  //glEnable(GL_CULL_FACE);
  //glCullFace(GL_BACK);
  //glFrontFace(GL_CCW);
  scene.ubo.compareSizeWithUniformBlock(assetManager.shaders.at("phongShadowed").getId(), "SceneUniforms");   

  while (!workshop.shouldStop()) {
    workshop.beginFrame();
    const glm::uvec2 winSize = workshop.getWindowSize();

    workshop.drawUI();

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
    if (ImGui::ColorEdit4("Shadow Border Color", shadowBorderColor))
      glTextureParameterfv(shadowFbo.getDepthAttachment().getId(), GL_TEXTURE_BORDER_COLOR, shadowBorderColor);
    static bool shouldShadowZeroOutsideFarPlane = true;
    ImGui::Checkbox("Shadow=0 outside far-plane", &shouldShadowZeroOutsideFarPlane);
    static bool shouldDoPcf = true;
    ImGui::Checkbox("Shadow PCF", &shouldDoPcf);
    glm::ivec2 shadowToggles{shouldShadowZeroOutsideFarPlane, shouldDoPcf};
    //static bool cullFrontFaces = false;
    //ImGui::Checkbox("Cull Front Faces", &cullFrontFaces);
    ImGui::Separator();
    ImGui::End();

    shadowFbo.resizeIfNeeded(light.shadowWidth, light.shadowHeight);
    orbitingCamController.update(workshop.getFrameDurationSec());
    cam.aspectRatio = static_cast<float>(winSize.x) / winSize.y;
    scene.uploadUniforms();

    glClearColor(bgColor.x, bgColor.y, bgColor.z, 1);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    auto drawScene = [&]() {
      glViewport(0, 0, winSize.x, winSize.y);
      for (auto& renderable : scene.renderables) {
        ws::Shader& shader = renderable.get().material.shader;
        shader.bind();
        renderable.get().material.uploadParameters();
        shadowFbo.getDepthAttachment().bindToUnit(1);
        shader.setMatrix4("u_LightSpaceMatrix", light.getLightSpaceMatrix());
        shader.setVector3("u_LightPos", light.position);
        shader.setFloat("u_LightIntensity", light.intensity);
        shader.setVector2("u_ShadowBias", light.shadowBias);
        shader.setIntVector2("u_ShadowToggles", shadowToggles);
        shader.setMatrix4("u_WorldFromObject", renderable.get().transform.getWorldFromObjectMatrix());
        // TODO: not there yet. Positions and scale inheritence looks fine, but rotation is broken. Parent's rotation should rotate child's coordinate system.
        //shader.setMatrix4("u_WorldFromObject", renderable.get().getGlobalTransformMatrix());
        renderable.get().mesh.draw();
        shader.unbind();
      }
    };

    auto drawShadowMap = [&]() {
      glViewport(0, 0, light.shadowWidth, light.shadowHeight);
      shadowFbo.bind();
      //if (cullFrontFaces) glCullFace(GL_FRONT);
      glClear(GL_DEPTH_BUFFER_BIT);
      assetManager.shaders.at("simpleDepth").bind();
      // cam.getProjectionFromView() * cam.getViewFromWorld() to see from camera's perspective
      assetManager.shaders.at("simpleDepth").setMatrix4("u_LightSpaceMatrix", light.getLightSpaceMatrix());

      for (auto& renderable : scene.renderables) {
        assetManager.shaders.at("simpleDepth").setMatrix4("u_WorldFromObject", renderable.get().transform.getWorldFromObjectMatrix());
        renderable.get().mesh.draw();
      }

      assetManager.shaders.at("simpleDepth").unbind();
      //if (cullFrontFaces) glCullFace(GL_BACK);
      shadowFbo.unbind();
    };

    auto visualizeDepth = [&]() {
      glViewport(0, 0, winSize.x, winSize.y);
      const ws::Shader& shader = assetManager.shaders.at("depthViz");
      shader.bind();

      shader.setFloat("near_plane", light.near);
      shader.setFloat("far_plane", light.far);
      glBindTextureUnit(0, shadowFbo.getDepthAttachment().getId());
      glBindVertexArray(dummyVao);
      glDrawArrays(GL_TRIANGLES, 0, 6);
      glBindVertexArray(0);
      shader.unbind();
    };

    drawShadowMap();
    drawScene();
    //visualizeDepth();

    textureViewer.draw();
    ws::VObjectPtr selectedObject = hierarchyWindow.draw({});
    inspectorWindow.inspectObject(selectedObject);

    workshop.endFrame();
  }

  std::println("Bye!");
  return 0;
}