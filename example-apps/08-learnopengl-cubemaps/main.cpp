#include <Workshop/AssetManager.hpp>
#include <Workshop/Assets.hpp>
#include <Workshop/Camera.hpp>
#include <Workshop/Cubemap.hpp>
#include <Workshop/Model.hpp>
#include <Workshop/Scene.hpp>
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

const std::filesystem::path SRC{SOURCE_DIR};

class Skybox {
 public:
  ws::Mesh mesh{ws::loadOBJ(ws::ASSETS_FOLDER / "models/cube.obj")};
  ws::Shader shader;
  ws::Cubemap cubemap;

  using path = std::filesystem::path;
  Skybox(const path& right, const path& left, const path& top, const path& bottom, const path& front, const path& back, bool optimized = true)
      : shader{[optimized]() { 
          if (optimized) return ws::Shader{ws::ASSETS_FOLDER / "shaders/skybox.vert", ws::ASSETS_FOLDER / "shaders/skybox.frag"}; 
          else return ws::Shader{SRC / "skybox_not_optimized.vert", SRC / "skybox_not_optimized.frag"};
        }()},
    cubemap{right, left, top, bottom, front, back} {}
};


void drawSkybox(const Skybox& skybox, const ws::Camera& cam);
void drawSceneWithCamera(const ws::Scene& scene, const ws::Camera& cam, const Skybox& skybox);

int main() {
  std::println("Hi!");
  ws::Workshop workshop{2048, 1536, "Workshop App"};

  ws::AssetManager assetManager;
  assetManager.meshes.emplace("cube", ws::loadOBJ(ws::ASSETS_FOLDER / "models/cube.obj"));
  assetManager.meshes.emplace("monkey", ws::loadOBJ(ws::ASSETS_FOLDER / "models/suzanne_smooth.obj"));
  assetManager.meshes.emplace("teapot", ws::loadOBJ(ws::ASSETS_FOLDER / "models/teapot-4k.obj"));
  assetManager.textures.emplace("container", ws::Texture{ws::ASSETS_FOLDER / "images/LearnOpenGL/container.jpg"});
  assetManager.textures.emplace("marble", ws::Texture{ws::ASSETS_FOLDER / "images/LearnOpenGL/marble.jpg"});
  assetManager.shaders.emplace("unlit", ws::Shader{ws::ASSETS_FOLDER / "shaders/unlit.vert", ws::ASSETS_FOLDER / "shaders/unlit.frag"});
  assetManager.shaders.emplace("reflective", ws::Shader{SRC / "reflective.vert", SRC / "reflective.frag"});
  assetManager.materials.emplace("unlit-box", ws::Material{
    .shader = assetManager.shaders.at("unlit"),
    .parameters = {
      {"mainTex", assetManager.textures.at("marble")},
      {"u_Color", glm::vec4(1, 1, 1, 1)},
    },
  });
  assetManager.materials.emplace("reflective-monkey", ws::Material{
    .shader = assetManager.shaders.at("reflective"),
    .parameters = {
      {"u_Color", glm::vec4(1, 1, 0, 1)},
      {"u_RefRefMix", 0.25f},
    },
    .shouldMatchUniforms = false,
  });
  assetManager.materials.emplace("reflective-teapot", ws::Material{
    .shader = assetManager.shaders.at("reflective"),
    .parameters = {
      {"u_Color", glm::vec4(0, 1, 1, 1)},
      {"u_RefRefMix", 0.75f},
    },
    .shouldMatchUniforms = false,
  });
  //assert(assetManager.doAllMaterialsHaveMatchingParametersAndUniforms());
  const std::filesystem::path base = ws::ASSETS_FOLDER / "images/LearnOpenGL/skybox";
  Skybox skybox{base / "right.jpg", base / "left.jpg", base / "top.jpg", base / "bottom.jpg", base / "front.jpg", base / "back.jpg"};
  Skybox skyboxNotOptimized{base / "right.jpg", base / "left.jpg", base / "top.jpg", base / "bottom.jpg", base / "front.jpg", base / "back.jpg", false};

  ws::RenderableObject box = {
    {"Box", {glm::vec3{0, 0, 0}, glm::vec3{0, 0, 1}, 0, glm::vec3{2, 2, 2}}},
    assetManager.meshes.at("cube"),
    assetManager.materials.at("unlit-box"),
  };
  ws::RenderableObject glassyMonkey = {
    {"Monkey", {glm::vec3{3, 0, 0}, glm::vec3{0, 0, 1}, 0, glm::vec3{1, 1, 1}}},
    assetManager.meshes.at("monkey"),
    assetManager.materials.at("reflective-monkey"),
  };
  ws::RenderableObject glassyTeapot = {
    {"Teapot", {glm::vec3{0, 1.5, 0}, glm::vec3{0, 0, 1}, 0, glm::vec3{1, 1, 1}}},
    assetManager.meshes.at("teapot"),
    assetManager.materials.at("reflective-teapot"),
  };

  ws::Scene scene {
    .renderables = {box, glassyMonkey, glassyTeapot},
  };

  ws::AutoOrbitingCameraController orbitingCamController{scene.camera};
  orbitingCamController.radius = 13.8f;
  orbitingCamController.theta = 0.355f;
  orbitingCamController.speed = 0.5;

  glEnable(GL_DEPTH_TEST);
  scene.ubo.compareSizeWithUniformBlock(assetManager.shaders.at("unlit").getId(), "SceneUniforms");

  while (!workshop.shouldStop())
  {
    workshop.beginFrame();
    const glm::uvec2 winSize = workshop.getWindowSize();

    workshop.drawUI();

    ImGui::Begin("Main");
    static bool shouldOptimize = true;
    ImGui::Checkbox("Should Optimize?", &shouldOptimize);
    ImGui::End();

    orbitingCamController.update(0.01f);
    scene.camera.aspectRatio = static_cast<float>(winSize.x) / winSize.y;
    scene.uploadUniforms();

    glClearColor(1, 0, 1, 1);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // First draw skybox without writing into depth buffer
    // Then draw scene objects
    // Any pixel where a scene object appears will be overdrawn
    auto drawAllUnoptimized = [&]() {
      glDepthMask(GL_FALSE);
      drawSkybox(skyboxNotOptimized, scene.camera);
      glDepthMask(GL_TRUE);

      drawSceneWithCamera(scene, scene.camera, skyboxNotOptimized);
    };

    // First draw the scene, store depth
    // Then draw skybox only to pixels where no scene object has been drawn
    auto drawAll = [&]() {
      drawSceneWithCamera(scene, scene.camera, skybox);

      // At a pixel, when no scene object is drawn the cleared depth value is 1
      glDepthFunc(GL_LEQUAL);  // change depth function so depth test passes when values are equal to depth buffer's content, i.e. when 1, i.e. when no object drawn
      drawSkybox(skybox, scene.camera);
      glDepthFunc(GL_LESS); // back to default
    };

    ImGui::Begin("Materials");
    if (shouldOptimize)
      drawAll();
    else
      drawAllUnoptimized();
    ImGui::End();

    workshop.endFrame();
  }

  std::println("Bye!");
  return 0;
}

void drawSceneWithCamera(const ws::Scene& scene, const ws::Camera& cam, const Skybox& skybox) {
  for (auto& objRef : scene.renderables) {
    auto& obj = objRef.get();

    obj.material.shader.bind();
    obj.material.uploadParameters();
    glBindTextureUnit(1, skybox.cubemap.getId());
    // Camera uniforms
    obj.material.shader.setVector3("u_CameraPos", cam.position);
    obj.material.shader.setMatrix4("u_ViewFromWorld", cam.getViewFromWorld());
    obj.material.shader.setMatrix4("u_ProjectionFromView", cam.getProjectionFromView());
    // Object uniforms
    obj.material.shader.setMatrix4("u_WorldFromObject", obj.transform.getWorldFromObjectMatrix());

    obj.mesh.draw();

    glBindTextureUnit(0, 0);
    glBindTextureUnit(1, 0);
    obj.material.shader.unbind();
  }
}

void drawSkybox(const Skybox& skybox, const ws::Camera& cam) {
  skybox.shader.bind();
  // skyboxShader.setMatrix4("u_WorldFromObject", cube.transform.getWorldFromObjectMatrix());
  const glm::mat4 viewWithoutTranslation = ws::removeTranslation(cam.getViewFromWorld());
  skybox.shader.setMatrix4("u_ViewFromWorld", viewWithoutTranslation);
  skybox.shader.setMatrix4("u_ProjectionFromView", cam.getProjectionFromView());
  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_CUBE_MAP, skybox.cubemap.getId());
  skybox.mesh.draw();
  glBindTexture(GL_TEXTURE_CUBE_MAP, 0);
  skybox.shader.unbind();
}