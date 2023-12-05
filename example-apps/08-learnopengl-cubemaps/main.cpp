#include <Workshop/Assets.hpp>
#include <Workshop/Camera.hpp>
#include <Workshop/Cubemap.hpp>
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


struct Renderable {
  std::string name;
  ws::Mesh mesh;
  ws::Shader shader;
  ws::Transform transform;
  ws::Texture texture;
  glm::vec4 color{1, 1, 1, 1};
  float refRefMix = 0.25f;
};
using Scene = std::vector<std::reference_wrapper<Renderable>>;

void drawSkybox(const Skybox& skybox, const ws::ICamera& cam);
void drawSceneWithCamera(const Scene& scene, const ws::ICamera& cam, const Skybox& skybox);

int main() {
  std::println("Hi!");
  ws::Workshop workshop{2048, 1536, "Workshop App"};

  const std::filesystem::path base = ws::ASSETS_FOLDER / "images/LearnOpenGL/skybox";
  Skybox skybox{base / "right.jpg", base / "left.jpg", base / "top.jpg", base / "bottom.jpg", base / "front.jpg", base / "back.jpg"};
  Skybox skyboxNotOptimized{base / "right.jpg", base / "left.jpg", base / "top.jpg", base / "bottom.jpg", base / "front.jpg", base / "back.jpg", false};

  Renderable box {
    .name = "Box",
    .mesh{ws::loadOBJ(ws::ASSETS_FOLDER / "models/cube.obj")},
    .shader{ws::ASSETS_FOLDER / "shaders/unlit.vert", ws::ASSETS_FOLDER / "shaders/unlit.frag"},
    .transform{glm::vec3{0, 0, 0}, glm::vec3{0, 0, 1}, 0, glm::vec3{2, 2, 2}},
    .texture{ws::ASSETS_FOLDER / "images/LearnOpenGL/container.jpg"},
  };
  Renderable glassyMonkey {
    .name = "Monkey",
    .mesh{ws::loadOBJ(ws::ASSETS_FOLDER / "models/suzanne_smooth.obj")},
    .shader{SRC / "reflective.vert", SRC / "reflective.frag"},
    .transform{glm::vec3{3, 0, 0}, glm::vec3{0, 0, 1}, 0, glm::vec3{1, 1, 1}},
    .color{1, 1, 0, 1},
    .refRefMix{0.25},
  };
  Renderable glassyTeapot {
    .name = "Teapot",
    .mesh{ws::loadOBJ(ws::ASSETS_FOLDER / "models/teapot-4k.obj")},
    .shader{SRC / "reflective.vert", SRC / "reflective.frag"},
    .transform{glm::vec3{0, 1.5, 0}, glm::vec3{0, 0, 1}, 0, glm::vec3{1, 1, 1}},
    .color{0, 1, 1, 1},
    .refRefMix{0.75},
  };
  Scene renderables = {box, glassyMonkey, glassyTeapot};

  ws::PerspectiveCamera3D cam;
  ws::AutoOrbitingCamera3DViewController orbitingCamController{cam};
  orbitingCamController.radius = 13.8f;
  orbitingCamController.theta = 0.355f;

  glEnable(GL_DEPTH_TEST);

  while (!workshop.shouldStop())
  {
    workshop.beginFrame();
    const glm::uvec2 winSize = workshop.getWindowSize();

    workshop.imGuiDrawAppWindow();

    ImGui::Begin("Main");
    static bool shouldOptimize = true;
    ImGui::Checkbox("Should Optimize?", &shouldOptimize);
    ImGui::End();

    orbitingCamController.update(0.01f);
    cam.aspectRatio = static_cast<float>(winSize.x) / winSize.y;

    glClearColor(1, 0, 1, 1);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // First draw skybox without writing into depth buffer
    // Then draw scene objects
    // Any pixel where a scene object appears will be overdrawn
    auto drawAllUnoptimized = [&]() {
      glDepthMask(GL_FALSE);
      drawSkybox(skyboxNotOptimized, cam);
      glDepthMask(GL_TRUE);

      drawSceneWithCamera(renderables, cam, skyboxNotOptimized);
    };

    // First draw the scene, store depth
    // Then draw skybox only to pixels where no scene object has been drawn
    auto drawAll = [&]() {
      drawSceneWithCamera(renderables, cam, skybox);

      // At a pixel, when no scene object is drawn the cleared depth value is 1
      glDepthFunc(GL_LEQUAL);  // change depth function so depth test passes when values are equal to depth buffer's content, i.e. when 1, i.e. when no object drawn
      drawSkybox(skybox, cam);
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

void drawSceneWithCamera(const Scene& scene, const ws::ICamera& cam, const Skybox& skybox) {
  for (auto& objRef : scene) {
    auto& obj = objRef.get();

    obj.shader.bind();
    obj.mesh.bind();

    glBindTextureUnit(0, obj.texture.getId());
    glBindTextureUnit(1, skybox.cubemap.getId());
    // Camera uniforms
    obj.shader.setVector3("u_CameraPos", cam.getPosition());
    obj.shader.setMatrix4("u_ViewFromWorld", cam.getViewFromWorld());
    obj.shader.setMatrix4("u_ProjectionFromView", cam.getProjectionFromView());
    // Scene uniforms
    // Object uniforms
    obj.shader.setMatrix4("u_WorldFromObject", obj.transform.getWorldFromObjectMatrix());
    // Material uniforms
    ImGui::SliderFloat(std::format("{} Mix", obj.name).c_str(), &obj.refRefMix, 0, 1);
    obj.shader.setFloat("u_RefRefMix", obj.refRefMix);
    obj.shader.setVector4("u_Color", obj.color);

    obj.mesh.draw();

    glBindTextureUnit(0, 0);
    glBindTextureUnit(1, 0);
    obj.mesh.unbind();
    obj.shader.unbind();
  }
}

void drawSkybox(const Skybox& skybox, const ws::ICamera& cam) {
  skybox.shader.bind();
  // skyboxShader.setMatrix4("u_WorldFromObject", cube.transform.getWorldFromObjectMatrix());
  const glm::mat4 viewWithoutTranslation = ws::removeTranslation(cam.getViewFromWorld());
  skybox.shader.setMatrix4("u_ViewFromWorld", viewWithoutTranslation);
  skybox.shader.setMatrix4("u_ProjectionFromView", cam.getProjectionFromView());
  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_CUBE_MAP, skybox.cubemap.getId());
  skybox.mesh.bind();
  skybox.mesh.draw();
  skybox.mesh.unbind();
  glBindTexture(GL_TEXTURE_CUBE_MAP, 0);
  skybox.shader.unbind();
}