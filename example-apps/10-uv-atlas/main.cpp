#include "LightMapper.hpp"

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
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>
#include <xatlas.h>

#include <print>
#include <ranges>
#include <string>
#include <vector>

const std::filesystem::path SRC{SOURCE_DIR};

class AssetManager {
 public:
  std::unordered_map<std::string, ws::Mesh> meshes;
  std::unordered_map<std::string, ws::Texture> textures;
};

int main() {
  std::println("Hi!");
  ws::Workshop workshop{1920, 1080, "UV Atlas Generation - Lightmapping"};

  AssetManager assetManager;
  assetManager.meshes.emplace("monkey1", ws::loadOBJ(ws::ASSETS_FOLDER / "models/suzanne.obj"));
  assetManager.meshes.emplace("monkey2", ws::loadOBJ(ws::ASSETS_FOLDER / "models/suzanne.obj"));
  assetManager.meshes.emplace("cube1", ws::loadOBJ(ws::ASSETS_FOLDER / "models/cube.obj"));
  assetManager.meshes.emplace("cube2", ws::loadOBJ(ws::ASSETS_FOLDER / "models/cube.obj"));
  assetManager.meshes.emplace("torus", ws::loadOBJ(ws::ASSETS_FOLDER / "models/torus.obj"));
  assetManager.meshes.emplace("baked_scene", ws::loadOBJ(SRC / "baked_scene.obj"));
  assetManager.textures.emplace("uv_grid", ws::ASSETS_FOLDER / "images/Wikipedia/UV_checker_Map_byValle.jpg");
  assetManager.textures.emplace("wood", ws::ASSETS_FOLDER / "images/LearnOpenGL/container.jpg");
  assetManager.textures.emplace("metal", ws::ASSETS_FOLDER / "images/LearnOpenGL/metal.png");
  assetManager.textures.emplace("baked_lightmap", SRC / "baked_lightmap.png");
  ws::Texture whiteTex{ws::Texture::Specs{1, 1, ws::Texture::Format::RGB8, ws::Texture::Filter::Linear}};
  std::vector<uint32_t> whiteTexPixels = {0xFFFFFF};
  whiteTex.uploadPixels(whiteTexPixels.data());
  assetManager.textures.emplace("white", std::move(whiteTex));
  ws::Shader mainShader{ws::ASSETS_FOLDER / "shaders/phong.vert", ws::ASSETS_FOLDER / "shaders/phong.frag"};
  ws::Shader unlitShader{ws::ASSETS_FOLDER / "shaders/unlit.vert", ws::ASSETS_FOLDER / "shaders/unlit.frag"};
  ws::Shader debugShader{ws::ASSETS_FOLDER / "shaders/debug.vert", ws::ASSETS_FOLDER / "shaders/debug.frag"};
  ws::Shader uvAtlasShader{SRC / "uv_atlas.vert", SRC / "uv_atlas.frag"};
  ws::Shader lightmapShader{SRC / "lightmap.vert", SRC / "lightmap.frag"};
  ws::Framebuffer atlasFbo = ws::Framebuffer::makeDefaultColorOnly(1, 1);

  const bool shouldUseLightmap = true;
  ws::Shader& objShader = shouldUseLightmap ? lightmapShader : mainShader;
  ws::Texture& obj2ndTex = shouldUseLightmap ? assetManager.textures.at("baked_lightmap") : whiteTex;
  ws::RenderableObject ground = {
      {"Ground", {glm::vec3{0, -1, 0}, glm::vec3{0, 0, 1}, 0, glm::vec3{20.f, .1f, 20.f}}},
      assetManager.meshes.at("cube1"),
      objShader,
      assetManager.textures["wood"],
      obj2ndTex,
  };
  ws::RenderableObject monkey1 = {
      {"Monkey1", {glm::vec3{0, -.15f, 0}, glm::vec3{1, 0, 0}, glm::radians(-30.f), glm::vec3{1.5f, 1.5f, 1.5f}}},
      assetManager.meshes.at("monkey1"),
      objShader,
      assetManager.textures["uv_grid"],
      obj2ndTex,
  };
  ws::RenderableObject monkey2 = {
      {"Monkey2", {glm::vec3{4, 0, 1}, glm::vec3{0, 1, 0}, glm::radians(55.f), glm::vec3{1.f, 1.f, 1.f}}},
      assetManager.meshes.at("monkey2"),
      objShader,
      assetManager.textures["wood"],
      obj2ndTex,
  };
  ws::RenderableObject box = {
      {"Box", {glm::vec3{1.6f, 0, 2.2f}, glm::vec3{0, 1, 0}, glm::radians(-22.f), glm::vec3{1.f, 2.f, 2.f}}},
      assetManager.meshes.at("cube2"),
      objShader,
      assetManager.textures["wood"],
      obj2ndTex,
  };
  ws::RenderableObject torus = {
      {"Torus", {glm::vec3{1.5, 2, 3}, glm::vec3{0, 1, 1}, glm::radians(30.f), glm::vec3{1.f, 1.f, 1.f}}},
      assetManager.meshes.at("torus"),
      objShader,
      assetManager.textures["metal"],
      obj2ndTex,
  };
  ws::RenderableObject bakedScene = {
      {"BakedScene", {glm::vec3{0, 0, 0}, glm::vec3{0, 1, 0}, 0, glm::vec3{1.f, 1.f, 1.f}}},
      assetManager.meshes.at("baked_scene"),
      unlitShader,
      assetManager.textures["baked_lightmap"],
      whiteTex,
  };
  ws::PerspectiveCamera3D cam;
  ws::Scene scene{
    .renderables{monkey1, monkey2, box, torus, ground},
    //.renderables{bakedScene},
  };
  ws::setParent(&ground, &scene.root);
  ws::setParent(&monkey1, &scene.root);
  ws::setParent(&monkey2, &scene.root);
  ws::setParent(&box, &scene.root);
  ws::setParent(&torus, &scene.root);

  LightMapper lightMapper; 
  lightMapper.generateUV2Atlas(scene);

  ws::AutoOrbitingCamera3DViewController orbitingCamController{cam};
  orbitingCamController.radius = 10.f;
  orbitingCamController.theta = 0.3f;
  const std::vector<std::reference_wrapper<ws::Texture>> texRefs{atlasFbo.getFirstColorAttachment(), assetManager.textures.at("baked_lightmap")};
  ws::TextureViewer textureViewer{texRefs};
  ws::HierarchyWindow hierarchyWindow{scene};
  ws::InspectorWindow inspectorWindow{};
  workshop.shadersToReload = {mainShader, unlitShader, debugShader, uvAtlasShader, lightmapShader};
  
  glEnable(GL_DEPTH_TEST);
  
  while (!workshop.shouldStop()) {
    workshop.beginFrame();
    const glm::uvec2 winSize = workshop.getWindowSize();
	const glm::uvec2 atlasSize = lightMapper.getAtlasSize();
	atlasFbo.resizeIfNeeded(atlasSize.x, atlasSize.y);

    ImGui::Begin("LightMapper");
    static glm::vec3 bgColor{42 / 256.0, 96 / 256.0, 87 / 256.0};
    static bool debugScene = false;
    ImGui::Checkbox("Debug Scene using debug shader", &debugScene);
    ImGui::ColorEdit3("BG Color", glm::value_ptr(bgColor));
    ImGui::Separator();

    if (ImGui::Button("Save my Uv Atlas Image")) {
      const ws::Texture& tex = atlasFbo.getFirstColorAttachment();
      const uint32_t w = tex.specs.width, h = tex.specs.height;
      uint32_t* pixels = new uint32_t[w * h];
      glGetTextureSubImage(tex.getId(), 0, 0, 0, 0, w, h, 1, GL_RGBA, GL_UNSIGNED_BYTE, sizeof(uint32_t) * w * h, pixels);
      stbi_write_png("uv_atlas_vug.png", w, h, 4, pixels, sizeof(uint32_t) * w);
      delete[] pixels;
    }
    ImGui::End();

    orbitingCamController.update(workshop.getFrameDurationMs() * 0.001f);
    cam.aspectRatio = static_cast<float>(winSize.x) / winSize.y;

    atlasFbo.bind();
    glViewport(0, 0, atlasSize.x, atlasSize.y);
    glDisable(GL_CULL_FACE);
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    glClearColor(0, 0, 0, 0);
    glClear(GL_COLOR_BUFFER_BIT);
    uvAtlasShader.bind();
    uvAtlasShader.setMatrix4("u_WorldFromObject", glm::mat4(1));
    uvAtlasShader.setMatrix4("u_ViewFromWorld", cam.getViewFromWorld());
    uvAtlasShader.setMatrix4("u_ProjectionFromView", cam.getProjectionFromView());
    for (auto& renderable : scene.renderables) {
      glBindTextureUnit(0, renderable.get().texture.getId());
      const ws::Mesh& mesh = renderable.get().mesh;
      mesh.bind();
      mesh.draw();
      mesh.unbind();
    }
    uvAtlasShader.unbind();
    atlasFbo.unbind();

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
      shader.setVector3("u_CameraPosition", cam.getPosition());
      if (debugScene)
        shader.setVector2("u_CameraNearFar", glm::vec2{cam.nearClip, cam.farClip});
      glBindTextureUnit(0, renderable.get().texture.getId());
      glBindTextureUnit(1, renderable.get().texture2.getId());
      shader.setMatrix4("u_WorldFromObject", renderable.get().transform.getWorldFromObjectMatrix());
      renderable.get().mesh.bind();
      renderable.get().mesh.draw();
      renderable.get().mesh.unbind();
      glBindTextureUnit(0, 0);
      glBindTextureUnit(1, 0);
      shader.unbind();
    }

	workshop.drawUI();
    textureViewer.draw();
    ws::VObjectPtr selectedObject = hierarchyWindow.draw();
    inspectorWindow.inspectObject(selectedObject);
	lightMapper.drawUI(scene);

    workshop.endFrame();
  }

  std::println("Bye!");
  return 0;
}