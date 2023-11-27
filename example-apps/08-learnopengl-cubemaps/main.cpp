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

class Skybox {
 public:
  ws::Mesh mesh{ws::loadOBJ(ws::ASSETS_FOLDER / "models/cube.obj")};
  ws::Shader shader{ws::ASSETS_FOLDER / "shaders/skybox.vert", ws::ASSETS_FOLDER / "shaders/skybox.frag"};
  ws::Cubemap cubemap;

  using path = std::filesystem::path;
  Skybox(const path& right, const path& left, const path& top, const path& bottom, const path& front, const path& back)
      : cubemap{right, left, top, bottom, front, back} {}
};

struct Renderable {
  ws::Mesh mesh;
  ws::Shader shader;
  ws::Transform transform;
  ws::Texture texture;
};
using Scene = std::vector<std::reference_wrapper<Renderable>>;

void drawSkybox(const Skybox& skybox, const ws::ICamera& cam);
void drawSceneWithCamera(const Scene& scene, const ws::ICamera& cam);

int main()
{
  std::println("Hi!");
  ws::Workshop workshop{2048, 1536, "Workshop App"};

  const std::filesystem::path base = ws::ASSETS_FOLDER / "images/LearnOpenGL/skybox";
  Skybox skybox{base / "right.jpg", base / "left.jpg", base / "top.jpg", base / "bottom.jpg", base / "front.jpg", base / "back.jpg"};

  Renderable box{
    .mesh{ws::loadOBJ(ws::ASSETS_FOLDER / "models/cube.obj")},
    .shader{ws::ASSETS_FOLDER / "shaders/unlit.vert", ws::ASSETS_FOLDER / "shaders/unlit.frag"},
    .transform{glm::vec3{0, 0, 0}, glm::vec3{0, 0, 1}, 0, glm::vec3{2, 2, 2}},
    .texture{ws::ASSETS_FOLDER / "images/LearnOpenGL/container.jpg"},
  };
  Scene renderables = {box};
  

  ws::PerspectiveCamera3D cam;
  ws::AutoOrbitingCamera3DViewController orbitingCamController{cam};
  orbitingCamController.radius = 13.8f;
  orbitingCamController.theta = 0.355f;

  glEnable(GL_DEPTH_TEST);

  while (!workshop.shouldStop())
  {
    workshop.beginFrame();
    const glm::uvec2 winSize = workshop.getWindowSize();

    ImGui::Begin("Main");
    static bool shouldShowImGuiDemo = false;
    ImGui::Checkbox("Show Demo", &shouldShowImGuiDemo);
    if (shouldShowImGuiDemo)
      ImGui::ShowDemoWindow();

    static glm::vec3 bgColor{42 / 256.0, 96 / 256.0, 87 / 256.0};
    ImGui::ColorEdit3("BG Color", glm::value_ptr(bgColor));
    ImGui::End();

    orbitingCamController.update(0.01f);
    cam.aspectRatio = static_cast<float>(winSize.x) / winSize.y;

    glClearColor(bgColor.x, bgColor.y, bgColor.z, 1);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Unoptimizied
    glDepthMask(GL_FALSE);
    drawSkybox(skybox, cam);
    glDepthMask(GL_TRUE);
    drawSceneWithCamera(renderables, cam);

    workshop.endFrame();
  }

  std::println("Bye!");
  return 0;
}

void drawSceneWithCamera(const Scene& scene, const ws::ICamera& cam) {
  for (auto& objRef : scene) {
    const auto& obj = objRef.get();

    obj.shader.bind();
    obj.mesh.bind();

    ws::Texture::activateTexture(0);
    obj.texture.bind();
    obj.shader.setMatrix4("u_WorldFromObject", obj.transform.getWorldFromObjectMatrix());
    obj.shader.setMatrix4("u_ViewFromWorld", cam.getViewFromWorld());
    obj.shader.setMatrix4("u_ProjectionFromView", cam.getProjectionFromView());

    obj.mesh.draw();

    obj.texture.unbind();
    obj.mesh.unbind();
    obj.shader.unbind();
  }
}

void drawSkybox(const Skybox& skybox, const ws::ICamera& cam) {
  skybox.shader.bind();
  // skyboxShader.setMatrix4("u_WorldFromObject", cube.transform.getWorldFromObjectMatrix());
  const glm::mat4 viewWithoutTranslation = glm::mat4(glm::mat3(cam.getViewFromWorld()));
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