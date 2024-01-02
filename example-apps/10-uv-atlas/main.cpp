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
#include <vector>

const std::filesystem::path SRC{SOURCE_DIR};

class AssetManager {
 public:
  std::unordered_map<std::string, ws::Mesh> meshes;
  std::unordered_map<std::string, ws::Texture> textures;
};

int main()
{
  std::println("Hi!");
  ws::Workshop workshop{1024, 1024, "UV Atlas Generator"};

  ws::Shader debugShader{SRC / "debug.vert", SRC / "debug.frag"};
  ws::PerspectiveCamera3D cam;
  ws::AutoOrbitingCamera3DViewController orbitingCamController{cam};

  AssetManager assetManager;
  assetManager.meshes.emplace("monkey", ws::loadOBJ(ws::ASSETS_FOLDER / "models/suzanne.obj"));
  assetManager.meshes.emplace("cube", ws::loadOBJ(ws::ASSETS_FOLDER / "models/cube.obj"));
  assetManager.meshes.emplace("torus", ws::loadOBJ(ws::ASSETS_FOLDER / "models/torus.obj"));
  assetManager.textures.emplace("uv_grid", ws::ASSETS_FOLDER / "images/Wikipedia/UV_checker_Map_byValle.jpg");
  assetManager.textures.emplace("wood", ws::ASSETS_FOLDER / "images/LearnOpenGL/container.jpg");
  assetManager.textures.emplace("metal", ws::ASSETS_FOLDER / "images/LearnOpenGL/metal.png");
  ws::Texture whiteTex{ws::Texture::Specs{1, 1, ws::Texture::Format::RGB8, ws::Texture::Filter::Linear}};
  std::vector<uint32_t> whiteTexPixels = {0xFFFFFF};
  whiteTex.loadPixels(whiteTexPixels.data());
  assetManager.textures.emplace("white", std::move(whiteTex));
  ws::Shader mainShader{ws::ASSETS_FOLDER / "shaders/phong.vert", ws::ASSETS_FOLDER / "shaders/phong.frag"};

  ws::RenderableObject ground = {
      {"Ground", {glm::vec3{0, -1, 0}, glm::vec3{0, 0, 1}, 0, glm::vec3{10.f, 0.1f, 10.f}}},
      assetManager.meshes.at("cube"),
      mainShader,
      assetManager.textures["metal"],
  };
  ws::RenderableObject monkey1 = {
      {"Monkey1", {glm::vec3{0, 0, 0}, glm::vec3{0, 0, 1}, 0, glm::vec3{1.f, 1.f, 1.f}}},
      assetManager.meshes.at("monkey"),
      mainShader,
      assetManager.textures["uv_grid"],
  };
  ws::RenderableObject monkey2 = {
      {"Monkey2", {glm::vec3{3, 0, 0}, glm::vec3{0, 0, 1}, 0, glm::vec3{1.f, 1.f, 1.f}}},
      assetManager.meshes.at("monkey"),
      mainShader,
      assetManager.textures["wood"],
  };
  ws::Scene scene{
    .renderables{ground, monkey1, monkey2},
  };

  uint32_t numMeshes = 1;

  xatlas::Atlas* atlas = xatlas::Create();
  
  uint32_t totalVertices = 0;
  uint32_t totalFaces = 0;
  // xatlas::SetProgressCallback(atlas, ProgressCallback, &stopwatch);

  ws::Mesh& cube1 = assetManager.meshes.at("monkey");

  xatlas::MeshDecl meshDecl;
  std::vector<float> positions;
  std::vector<float> normals;
  std::vector<float> texCoords;
  for (const auto& v : cube1.meshData.vertices) {
    positions.push_back(v.position.x);
    positions.push_back(v.position.y);
    positions.push_back(v.position.z);
    normals.push_back(v.normal.x);
    normals.push_back(v.normal.y);
    normals.push_back(v.normal.z);
    texCoords.push_back(v.texCoord.x);
    texCoords.push_back(v.texCoord.y);
  }
  meshDecl.vertexCount = static_cast<uint32_t>(cube1.meshData.vertices.size());
  meshDecl.vertexPositionData = positions.data();
  meshDecl.vertexPositionStride = sizeof(float) * 3;
  meshDecl.vertexNormalData = normals.data();
  meshDecl.vertexNormalStride = sizeof(float) * 3;
  meshDecl.vertexUvData = texCoords.data();
  meshDecl.vertexUvStride = sizeof(float) * 2;
  meshDecl.indexCount = static_cast<uint32_t>(cube1.meshData.indices.size());
  meshDecl.indexData = cube1.meshData.indices.data();
  meshDecl.indexFormat = xatlas::IndexFormat::UInt32;

  xatlas::AddMeshError error = xatlas::AddMesh(atlas, meshDecl, numMeshes);
  if (error != xatlas::AddMeshError::Success) {
    xatlas::Destroy(atlas);
    std::println("rError adding mesh {}", xatlas::StringForEnum(error));
    return EXIT_FAILURE;
  }
  totalVertices += meshDecl.vertexCount;
  if (meshDecl.faceCount > 0)
    totalFaces += meshDecl.faceCount;
  else
    totalFaces += meshDecl.indexCount / 3;  // Assume triangles if MeshDecl::faceCount not specified.

  std::println("   {} total vertices", totalVertices);
  std::println("   {} total faces", totalFaces);
  // Generate atlas.
  std::println("Generating atlas");
  xatlas::Generate(atlas);

  //totalVertices = 0;
  //for (uint32_t i = 0; i < atlas->meshCount; i++) {
  //  const xatlas::Mesh& mesh = atlas->meshes[i];
  //  totalVertices += mesh.vertexCount;
  //  // Input and output index counts always match.
  //  assert(mesh.indexCount == static_cast<uint32_t>(cube1.meshData.indices.size())); // actually compare with meshes vector
  //}
  //std::println("   {} total vertices", totalVertices);
  //std::println("%.2f seconds (%g ms) elapsed total\n", globalStopwatch.elapsed() / 1000.0, globalStopwatch.elapsed());

  auto copyUV2 = [](xatlas::Atlas* atlas, ws::Mesh& sceneMesh) {
    const xatlas::Mesh& mesh = atlas->meshes[0];
    for (size_t ix = 0; ix < mesh.vertexCount; ++ix) {
      const xatlas::Vertex& v = mesh.vertexArray[ix];
      sceneMesh.meshData.vertices[v.xref].texCoord2 = {v.uv[0] / atlas->width, v.uv[1] / atlas->height};
    }
    sceneMesh.uploadData();
  };
  copyUV2(atlas, cube1);
  ws::Framebuffer atlasFbo = ws::Framebuffer::makeDefaultColorOnly(1, 1);

  const std::vector<std::reference_wrapper<ws::Texture>> texRefs{atlasFbo.getFirstColorAttachment()};
  ws::TextureViewer textureViewer{texRefs};
  
  glEnable(GL_DEPTH_TEST);
  
  while (!workshop.shouldStop()) {
    workshop.beginFrame();
    const glm::uvec2& winSize = workshop.getWindowSize();
    atlasFbo.resizeIfNeeded(atlas->width, atlas->height);

    workshop.imGuiDrawAppWindow();

    ImGui::Begin("Main");
    static glm::vec3 bgColor{42 / 256.0, 96 / 256.0, 87 / 256.0};
    ImGui::ColorEdit3("BG Color", glm::value_ptr(bgColor));
    ImGui::Separator();
    if (ImGui::Button("Recalculate UV Atlas and Upload to UV2s")) {
      xatlas::Generate(atlas);
      copyUV2(atlas, cube1);
    }
    ImGui::Text("Atlas Info");
    ImGui::Text("Size: (%d, %d)", atlas->width, atlas->height);
    ImGui::Text("#charts: %d, #atlases: %d", atlas->chartCount, atlas->atlasCount);
    for (uint32_t i = 0; i < atlas->atlasCount; i++)
      ImGui::Text("Atlas utilization: atlas[%d]: %.2f", i, atlas->utilization[i] * 100.0f);
    const ImVec2 tableOuterSize{0.f, 200.f};
    if (ImGui::BeginTable("Vertices", 6, ImGuiTableFlags_Borders | ImGuiTableFlags_SizingFixedFit | ImGuiTableFlags_ScrollY, tableOuterSize)) {
      ImGui::TableSetupColumn("origIx");
      ImGui::TableSetupColumn("atlasIx");
      ImGui::TableSetupColumn("texCoord");
      ImGui::TableSetupColumn("uv");
      ImGui::TableSetupColumn("atlas");
      ImGui::TableSetupColumn("chart");
      ImGui::TableHeadersRow();
      for (size_t ix = 0; ix < atlas->meshes[0].vertexCount; ++ix) {
        const xatlas::Vertex& v = atlas->meshes[0].vertexArray[ix];
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::Text("%3d", v.xref);        
        ImGui::TableNextColumn();
        ImGui::Text("%3d", ix);
        ImGui::TableNextColumn();
        ImGui::Text("(%6.1f, %6.1f)", v.uv[0], v.uv[1]);
        ImGui::TableNextColumn();
        ImGui::Text("(%1.3f, %1.3f)", v.uv[0] / atlas->width, v.uv[1] / atlas->height);
        ImGui::TableNextColumn();
        ImGui::Text("%d", v.atlasIndex);
        ImGui::TableNextColumn();
        ImGui::Text("%d", v.chartIndex);
      }
      ImGui::EndTable();
    }
    ImGui::End();

    orbitingCamController.update(workshop.getFrameDurationMs() * 0.001f);
    cam.aspectRatio = static_cast<float>(winSize.x) / winSize.y;

    atlasFbo.bind();
    glViewport(0, 0, atlas->width, atlas->height);
    glDisable(GL_CULL_FACE);
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    glClearColor(0, 0, 0, 0);
    glClear(GL_COLOR_BUFFER_BIT);
    debugShader.bind();
    debugShader.setMatrix4("u_WorldFromObject", glm::mat4(1));
    debugShader.setMatrix4("u_ViewFromWorld", cam.getViewFromWorld());
    debugShader.setMatrix4("u_ProjectionFromView", cam.getProjectionFromView());
    cube1.bind();
    cube1.draw();
    cube1.unbind();
    debugShader.unbind();
    atlasFbo.unbind();

    glViewport(0, 0, winSize.x, winSize.y);
    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    glClearColor(bgColor.x, bgColor.y, bgColor.z, 1);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    mainShader.bind();
    mainShader.setMatrix4("u_ViewFromWorld", cam.getViewFromWorld());
    mainShader.setMatrix4("u_ProjectionFromView", cam.getProjectionFromView());
    mainShader.setVector3("u_CameraPosition", cam.getPosition());
    glBindTextureUnit(1, assetManager.textures.at("white").getId());
    for (auto& renderable : scene.renderables) {
      mainShader.setMatrix4("u_WorldFromObject", renderable.get().transform.getWorldFromObjectMatrix());
      glBindTextureUnit(0, renderable.get().texture.getId());
      renderable.get().mesh.bind();
      renderable.get().mesh.draw();
      renderable.get().mesh.unbind();
    }
    mainShader.unbind();

    textureViewer.draw();
    workshop.endFrame();
  }

  xatlas::Destroy(atlas);
  std::println("Bye!");
  return 0;
}