#include <Workshop/Assets.hpp>
#include <Workshop/Camera.hpp>
#include <Workshop/Framebuffer.hpp>
#include <Workshop/Model.hpp>
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

int main()
{
  std::println("Hi!");
  ws::Workshop workshop{1024, 1024, "UV Atlas Generator"};

  ws::Shader debugShader{SRC / "debug.vert", SRC / "debug.frag"};
  ws::PerspectiveCamera3D cam;
  ws::AutoOrbitingCamera3DViewController orbitingCamController{cam};

  ws::Mesh cube1{ws::loadOBJ(ws::ASSETS_FOLDER / "models/suzanne.obj")};
  ws::Shader solidColorShader{ws::ASSETS_FOLDER / "shaders/phong.vert", ws::ASSETS_FOLDER / "shaders/phong.frag"};
  uint32_t numMeshes = 1;

  xatlas::Atlas* atlas = xatlas::Create();
  
  uint32_t totalVertices = 0;
  uint32_t totalFaces = 0;
  // xatlas::SetProgressCallback(atlas, ProgressCallback, &stopwatch);

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

  const xatlas::Mesh& mesh = atlas->meshes[0];
  for (size_t ix = 0; ix < mesh.vertexCount; ++ix) {
    const xatlas::Vertex& v = mesh.vertexArray[ix];
    cube1.meshData.vertices[v.xref].texCoord2 = {v.uv[0] / atlas->width, v.uv[1] / atlas->height};
  }
  cube1.uploadData();
  ws::Framebuffer atlasFbo = ws::Framebuffer::makeDefaultColorOnly(atlas->width, atlas->height);

  const std::vector<std::reference_wrapper<ws::Texture>> texRefs{atlasFbo.getFirstColorAttachment()};
  ws::TextureViewer textureViewer{texRefs};
  
  glEnable(GL_DEPTH_TEST);
  
  while (!workshop.shouldStop()) {
    workshop.beginFrame();
    const glm::uvec2& winSize = workshop.getWindowSize();

    workshop.imGuiDrawAppWindow();

    ImGui::Begin("Main");
    static glm::vec3 bgColor{42 / 256.0, 96 / 256.0, 87 / 256.0};
    ImGui::ColorEdit3("BG Color", glm::value_ptr(bgColor));
    ImGui::Separator();
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
      for (size_t ix = 0; ix < mesh.vertexCount; ++ix) {
        const xatlas::Vertex& v = mesh.vertexArray[ix];
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
    cube1.draw();
    debugShader.unbind();
    atlasFbo.unbind();

    glViewport(0, 0, winSize.x, winSize.y);
    glEnable(GL_CULL_FACE);
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    glCullFace(GL_BACK);
    glClearColor(bgColor.x, bgColor.y, bgColor.z, 1);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    solidColorShader.bind();
    solidColorShader.setMatrix4("u_WorldFromObject", glm::mat4(1));
    solidColorShader.setMatrix4("u_ViewFromWorld", cam.getViewFromWorld());
    solidColorShader.setMatrix4("u_ProjectionFromView", cam.getProjectionFromView());
    solidColorShader.setVector3("u_CameraPosition", cam.getPosition());
    cube1.draw();
    solidColorShader.unbind();

    textureViewer.draw();
    workshop.endFrame();
  }

  xatlas::Destroy(atlas);
  std::println("Bye!");
  return 0;
}