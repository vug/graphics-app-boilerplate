#include <Workshop/Assets.hpp>
#include <Workshop/Camera.hpp>
#include <Workshop/Cubemap.hpp>
#include <Workshop/Model.hpp>
#include <Workshop/Shader.hpp>
#include <Workshop/Workshop.hpp>

#include <glad/gl.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <glm/gtc/type_ptr.hpp>
#include <glm/vec3.hpp>

#include <print>

int main()
{
  std::println("Hi!");
  ws::Workshop workshop{800, 600, "Workshop App"};

  const std::filesystem::path base = ws::ASSETS_FOLDER / "images/LearnOpenGL/skybox";
  ws::Cubemap cubemap{base / "right.jpg", base / "left.jpg", base / "top.jpg", base / "bottom.jpg", base / "front.jpg", base / "back.jpg"};
  ws::Shader skyboxShader{ws::ASSETS_FOLDER / "shaders/skybox.vert", ws::ASSETS_FOLDER / "shaders/skybox.frag"};
  ws::Mesh cube{ws::loadOBJ(ws::ASSETS_FOLDER / "models/cube.obj")};

  ws::PerspectiveCamera3D cam;
  ws::AutoOrbitingCamera3DViewController orbitingCamController{cam};
  orbitingCamController.radius = 13.8f;
  orbitingCamController.theta = 0.355f;

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

    glDisable(GL_DEPTH_TEST);
    skyboxShader.bind();
    //skyboxShader.setMatrix4("u_WorldFromObject", cube.transform.getWorldFromObjectMatrix());
    const glm::mat4 viewWithoutTranslation = glm::mat4(glm::mat3(cam.getViewFromWorld()));
    skyboxShader.setMatrix4("u_ViewFromWorld", viewWithoutTranslation);
    skyboxShader.setMatrix4("u_ProjectionFromView", cam.getProjectionFromView());
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_CUBE_MAP, cubemap.getId());
    cube.bind();
    cube.draw();
    cube.unbind();
    glBindTexture(GL_TEXTURE_CUBE_MAP, 0);
    skyboxShader.unbind();
    glEnable(GL_DEPTH_TEST);

    workshop.endFrame();
  }

  std::println("Bye!");
  return 0;
}