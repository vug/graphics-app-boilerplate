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
#include <xatlas.h>

#include <print>
#include <ranges>
#include <string>
#include <vector>

const std::filesystem::path SRC{SOURCE_DIR};

int main() {
  std::println("Hi!");
  ws::Workshop workshop{1920, 1080, "UV Atlas Generation - Lightmapping"};

  ws::AssetManager assetManager;
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
  ws::Shader uvAtlasShader{SRC / "uv_atlas.vert", SRC / "uv_atlas.frag"};
  ws::Shader lightmapShader{SRC / "lightmap.vert", SRC / "lightmap.frag"};
  assetManager.materials.emplace("phong-wood", ws::Material{
    .shader = mainShader, //assetManager.shaders.at("phong"),
    .parameters = {
      {"diffuseTexture", assetManager.textures.at("wood")},
      {"specularTexture", assetManager.white}
    }
  });
  assetManager.materials.emplace("lightmap-wood", ws::Material{
    .shader = lightmapShader,
    .parameters = {
      {"diffuseTex", assetManager.textures.at("wood")},
      {"lightmapTex", assetManager.textures.at("baked_lightmap")}
    }
  });
  assetManager.materials.emplace("phong-uv_grid", ws::Material{
    .shader = mainShader, //assetManager.shaders.at("phong"),
    .parameters = {
      {"diffuseTexture", assetManager.textures.at("uv_grid")},
      {"specularTexture", assetManager.white}
    }
  });
  assetManager.materials.emplace("lightmap-uv_grid", ws::Material{
    .shader = lightmapShader,
    .parameters = {
      {"diffuseTex", assetManager.textures.at("uv_grid")},
      {"lightmapTex", assetManager.textures.at("baked_lightmap")}
    }
  });
  assetManager.materials.emplace("phong-metal", ws::Material{
    .shader = mainShader, //assetManager.shaders.at("phong"),
    .parameters = {
      {"diffuseTexture", assetManager.textures.at("metal")},
      {"specularTexture", assetManager.white}
    }
  });
  assetManager.materials.emplace("lightmap-metal", ws::Material{
    .shader = lightmapShader,
    .parameters = {
      {"diffuseTex", assetManager.textures.at("metal")},
      {"lightmapTex", assetManager.textures.at("baked_lightmap")}
    }
  });
  assetManager.materials.emplace("unlit-baked_scene", ws::Material{
    .shader = unlitShader,
    .parameters = {
      {"mainTex", assetManager.textures.at("baked_lightmap")},
      {"u_Color", glm::vec4(1, 1, 1, 1)},
    }
  });
  assert(assetManager.doAllMaterialsHaveMatchingParametersAndUniforms());
  ws::Framebuffer atlasFbo = ws::Framebuffer::makeDefaultColorOnly(1, 1);

  // true: adds light from the lightmap to unlit. don't forget to load UV2's from Lightmapper Window
  // false: uses real-time lights
  const bool shouldUseLightmap = true;
  // To debug the generated lightmap. Uses single mesh scene and single lightmap texture.
  const bool shouldJustShowBakedLight = false;
  ws::RenderableObject ground = {
      {"Ground", {glm::vec3{0, -1, 0}, glm::vec3{0, 0, 1}, 0, glm::vec3{20.f, .1f, 20.f}}},
      assetManager.meshes.at("cube1"),
      assetManager.materials.at(shouldUseLightmap ? "lightmap-wood" : "phong-wood"),
      assetManager.white,
      assetManager.white,
  };
  ws::RenderableObject monkey1 = {
      {"Monkey1", {glm::vec3{0, -.15f, 0}, glm::vec3{1, 0, 0}, glm::radians(-30.f), glm::vec3{1.5f, 1.5f, 1.5f}}},
      assetManager.meshes.at("monkey1"),
      assetManager.materials.at(shouldUseLightmap ? "lightmap-uv_grid" : "phong-uv_grid"),
      assetManager.white,
      assetManager.white,
  };
  ws::RenderableObject monkey2 = {
      {"Monkey2", {glm::vec3{4, 0, 1}, glm::vec3{0, 1, 0}, glm::radians(55.f), glm::vec3{1.f, 1.f, 1.f}}},
      assetManager.meshes.at("monkey2"),
      assetManager.materials.at(shouldUseLightmap ? "lightmap-wood" : "phong-wood"),
      assetManager.white,
      assetManager.white,
  };
  ws::RenderableObject box = {
      {"Box", {glm::vec3{1.6f, 0, 2.2f}, glm::vec3{0, 1, 0}, glm::radians(-22.f), glm::vec3{1.f, 2.f, 2.f}}},
      assetManager.meshes.at("cube2"),
      assetManager.materials.at(shouldUseLightmap ? "lightmap-wood" : "phong-wood"),
      assetManager.white,
      assetManager.white,
  };
  ws::RenderableObject torus = {
      {"Torus", {glm::vec3{1.5, 2, 3}, glm::vec3{0, 1, 1}, glm::radians(30.f), glm::vec3{1.f, 1.f, 1.f}}},
      assetManager.meshes.at("torus"),
      assetManager.materials.at(shouldUseLightmap ? "lightmap-metal" : "phong-metal"),
      assetManager.white,
      assetManager.white,
  };
  ws::RenderableObject bakedScene = {
      {"BakedScene", {glm::vec3{0, 0, 0}, glm::vec3{0, 1, 0}, 0, glm::vec3{1.f, 1.f, 1.f}}},
      assetManager.meshes.at("baked_scene"),
      assetManager.materials.at("unlit-baked_scene"),
      assetManager.white,
      assetManager.white,
  };
  ws::Scene scene{
    .directionalLights = std::vector<ws::DirectionalLight>{
      ws::DirectionalLight{
        .position = glm::vec3(1, 1, 1),
        .intensity = 0.5f,
        .direction = glm::vec3(-1, -1, -1),
        .color = glm::vec3(1, 1, 1),
      },
    },
  };
  if (shouldJustShowBakedLight)
    scene.renderables = {bakedScene};
  else
    scene.renderables = {monkey1, monkey2, box, torus, ground};
  ws::setParent(&ground, &scene.root);
  ws::setParent(&monkey1, &scene.root);
  ws::setParent(&monkey2, &scene.root);
  ws::setParent(&box, &scene.root);
  ws::setParent(&torus, &scene.root);

  LightMapper lightMapper; 
  lightMapper.generateUV2Atlas(scene);

	scene.camera.position = { 0, 5, -10 };
	scene.camera.target = { 0, 0, 0 };
  ws::ManualCameraController manualCamController{scene.camera};
  const std::vector<std::reference_wrapper<ws::Texture>> texRefs{atlasFbo.getFirstColorAttachment(), assetManager.textures.at("baked_lightmap")};
  ws::TextureViewer textureViewer{texRefs};
  ws::HierarchyWindow hierarchyWindow{scene};
  ws::InspectorWindow inspectorWindow{};
  ws::EditorWindow editorWindow{scene};
  workshop.shadersToReload = {mainShader, unlitShader, uvAtlasShader, lightmapShader};
  
  glEnable(GL_DEPTH_TEST);
  scene.ubo.compareSizeWithUniformBlock(mainShader.getId(), "SceneUniforms");
  
  while (!workshop.shouldStop()) {
    workshop.beginFrame();
    const glm::uvec2 winSize = workshop.getWindowSize();
	  const glm::uvec2 atlasSize = lightMapper.getAtlasSize();
	  atlasFbo.resizeIfNeeded(atlasSize.x, atlasSize.y);

    ImGui::Begin("LightMapper");
    static glm::vec3 bgColor{42 / 256.0, 96 / 256.0, 87 / 256.0};
    ImGui::ColorEdit3("BG Color", glm::value_ptr(bgColor));
    ImGui::Separator();

    if (ImGui::Button("Save my Uv Atlas Image"))
      atlasFbo.getFirstColorAttachment().saveToImageFile("uv_atlas_vug.png");
    ImGui::End();

    manualCamController.update(ws::getMouseCursorPosition(), workshop.mouseState, workshop.getFrameDurationMs() * 0.001f);
    scene.camera.aspectRatio = static_cast<float>(winSize.x) / winSize.y;
    scene.uploadUniforms();

    atlasFbo.bind();
    glViewport(0, 0, atlasSize.x, atlasSize.y);
    glDisable(GL_CULL_FACE);
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    glClearColor(0, 0, 0, 0);
    glClear(GL_COLOR_BUFFER_BIT);
    uvAtlasShader.bind();
    uvAtlasShader.setMatrix4("u_WorldFromObject", glm::mat4(1));
    uvAtlasShader.setMatrix4("u_ViewFromWorld", scene.camera.getViewFromWorld());
    uvAtlasShader.setMatrix4("u_ProjectionFromView", scene.camera.getProjectionFromView());
    for (auto& renderable : scene.renderables) {
      glBindTextureUnit(0, renderable.get().texture.getId());
      const ws::Mesh& mesh = renderable.get().mesh;
      mesh.draw();
    }
    uvAtlasShader.unbind();
    atlasFbo.unbind();

    glViewport(0, 0, winSize.x, winSize.y);
    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    glClearColor(bgColor.x, bgColor.y, bgColor.z, 1);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    scene.draw();

	  workshop.drawUI();
	  lightMapper.drawUI(scene);
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