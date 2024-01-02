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
  ws::Shader debugShader{SRC / "debug.vert", SRC / "debug.frag"};
  ws::Framebuffer atlasFbo = ws::Framebuffer::makeDefaultColorOnly(1, 1);

  ws::RenderableObject ground = {
      {"Ground", {glm::vec3{0, -1, 0}, glm::vec3{0, 0, 1}, 0, glm::vec3{20.f, .1f, 20.f}}},
      assetManager.meshes.at("cube"),
      mainShader,
      assetManager.textures["white"],
      whiteTex,
  };
  ws::RenderableObject monkey1 = {
      {"Monkey1", {glm::vec3{0, -.15f, 0}, glm::vec3{1, 0, 0}, glm::radians(-30.f), glm::vec3{1.5f, 1.5f, 1.5f}}},
      assetManager.meshes.at("monkey"),
      mainShader,
      assetManager.textures["uv_grid"],
      whiteTex,
  };
  ws::RenderableObject monkey2 = {
      {"Monkey2", {glm::vec3{4, 0, 1}, glm::vec3{0, 1, 0}, glm::radians(55.f), glm::vec3{1.f, 1.f, 1.f}}},
      assetManager.meshes.at("monkey"),
      mainShader,
      assetManager.textures["wood"],
      whiteTex,
  };
  ws::RenderableObject box = {
      {"Box", {glm::vec3{1.6f, 0, 2.2f}, glm::vec3{0, 1, 0}, glm::radians(-22.f), glm::vec3{1.f, 2.f, 2.f}}},
      assetManager.meshes.at("cube"),
      mainShader,
      assetManager.textures["wood"],
      whiteTex,
  };
  ws::RenderableObject torus = {
      {"Torus", {glm::vec3{1.5, 2, 3}, glm::vec3{0, 1, 1}, glm::radians(30.f), glm::vec3{1.f, 1.f, 1.f}}},
      assetManager.meshes.at("torus"),
      mainShader,
      assetManager.textures["metal"],
      whiteTex,
  };
  ws::PerspectiveCamera3D cam;
  ws::Scene scene{
    .renderables{monkey1, monkey2, box, torus, ground},
  };
  ws::setParent(&ground, &scene.root);
  ws::setParent(&monkey1, &scene.root);
  ws::setParent(&monkey2, &scene.root);
  ws::setParent(&box, &scene.root);
  ws::setParent(&torus, &scene.root);

  uint32_t numMeshes = 1;

  xatlas::Atlas* atlas = xatlas::Create();
  
  auto calcXAtlasMeshDecl = [](const ws::Mesh& wsMesh, const ws::Transform& transform) {
  xatlas::MeshDecl meshDecl;
    
    size_t numVertices = wsMesh.meshData.vertices.size();
    float* positions = new float[numVertices * 3];
    float* normals = new float[numVertices * 3];
    float* texCoords = new float[numVertices * 2];
    for (const auto& [ix, v] : wsMesh.meshData.vertices | std::ranges::views::enumerate) {
      const bool useWorldSpace = true;
      if (useWorldSpace) {
        glm::vec3 worldPos = transform.getWorldFromObjectMatrix() * glm::vec4(v.position, 1);
        positions[3 * ix + 0] = worldPos.x;
        positions[3 * ix + 1] = worldPos.y;
        positions[3 * ix + 2] = worldPos.z;
        glm::vec3 worldNormal = glm::mat3(glm::transpose(glm::inverse(transform.getWorldFromObjectMatrix()))) * v.normal;
        normals[3 * ix + 0] = worldNormal.x;
        normals[3 * ix + 1] = worldNormal.y;
        normals[3 * ix + 2] = worldNormal.z;      
      } else {
        positions[3 * ix + 0] = v.position.x;
        positions[3 * ix + 1] = v.position.y;
        positions[3 * ix + 2] = v.position.z;
        normals[3 * ix + 0] = v.normal.x;
        normals[3 * ix + 1] = v.normal.y;
        normals[3 * ix + 2] = v.normal.z;      
  }
      texCoords[2 * ix + 0] = v.texCoord.x;
      texCoords[2 * ix + 1] = v.texCoord.y;
    }
    meshDecl.vertexCount = static_cast<uint32_t>(wsMesh.meshData.vertices.size());
    meshDecl.vertexPositionData = positions;
  meshDecl.vertexPositionStride = sizeof(float) * 3;
    meshDecl.vertexNormalData = normals;
  meshDecl.vertexNormalStride = sizeof(float) * 3;
    meshDecl.vertexUvData = texCoords;
  meshDecl.vertexUvStride = sizeof(float) * 2;
    meshDecl.indexCount = static_cast<uint32_t>(wsMesh.meshData.indices.size());
    meshDecl.indexData = wsMesh.meshData.indices.data();
  meshDecl.indexFormat = xatlas::IndexFormat::UInt32;
    return meshDecl;
  };

  uint32_t numMeshes = static_cast<uint32_t>(scene.renderables.size());
  uint32_t totalVertices = 0;
  uint32_t totalFaces = 0;

  // xatlas::SetProgressCallback(atlas, ProgressCallback, &stopwatch);
  for (auto& r : scene.renderables) {
    xatlas::MeshDecl meshDecl = calcXAtlasMeshDecl(r.get().mesh, r.get().transform);
    assert(meshDecl.indexCount == static_cast<uint32_t>(r.get().mesh.meshData.indices.size()));
  xatlas::AddMeshError error = xatlas::AddMesh(atlas, meshDecl, numMeshes);
  if (error != xatlas::AddMeshError::Success) {
    xatlas::Destroy(atlas);
    std::println("rError adding mesh {}", xatlas::StringForEnum(error));
    return EXIT_FAILURE;
  }
  totalVertices += meshDecl.vertexCount;
    totalFaces += meshDecl.faceCount > 0 ? meshDecl.faceCount : meshDecl.indexCount / 3; // Assume triangles if MeshDecl::faceCount not specified.
  }
  std::println("{} total vertices, {} total faces", totalVertices, totalFaces);

  std::println("   {} total vertices", totalVertices);
  std::println("   {} total faces", totalFaces);
  // Generate atlas.
  std::println("Generating atlas");
  xatlas::Generate(atlas);

  auto copyUV2 = [](xatlas::Atlas* atlas, uint32_t meshIx, ws::Mesh& sceneMesh) {
    const xatlas::Mesh& mesh = atlas->meshes[meshIx];
    for (size_t ix = 0; ix < mesh.vertexCount; ++ix) {
      const xatlas::Vertex& v = mesh.vertexArray[ix];
      sceneMesh.meshData.vertices[v.xref].texCoord2 = {v.uv[0] / atlas->width, v.uv[1] / atlas->height};
    }
    sceneMesh.uploadData();
  };
  for (uint32_t ix = 0; ix < atlas->meshCount; ++ix) {
    copyUV2(atlas, ix, scene.renderables[ix].get().mesh);
  }

  ws::AutoOrbitingCamera3DViewController orbitingCamController{cam};
  orbitingCamController.radius = 10.f;
  orbitingCamController.theta = 0.3f;
  const std::vector<std::reference_wrapper<ws::Texture>> texRefs{atlasFbo.getFirstColorAttachment()};
  ws::TextureViewer textureViewer{texRefs};
  ws::HierarchyWindow hierarchyWindow{scene};
  ws::InspectorWindow inspectorWindow{};
  
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
    static xatlas::ChartOptions chartOptions;
    ImGui::Text("Chart Options");
    uint32_t n0 = 0, n1 = 1, n2 = 2, n5 = 5, n64 = 64, n1000 = 1000;
    ImGui::SliderFloat("Max Chart Area", &chartOptions.maxChartArea, 0, 10);
    ImGui::SliderFloat("Max Chart Boundary Length", &chartOptions.maxBoundaryLength, 0, 50);
    ImGui::SliderFloat("normalDeviationWeight", &chartOptions.normalDeviationWeight, 0, 10);
    ImGui::SliderFloat("roundnessWeight", &chartOptions.roundnessWeight, 0, 10);
    ImGui::SliderFloat("straightnessWeight", &chartOptions.straightnessWeight, 0, 10);
    ImGui::SliderFloat("normalSeamWeight", &chartOptions.normalSeamWeight, 0, 10);
    ImGui::SliderFloat("textureSeamWeight", &chartOptions.textureSeamWeight, 0, 10);
    ImGui::SliderFloat("Max Cost", &chartOptions.maxCost, 0, 10);
    ImGui::SliderScalar("Max Iterations", ImGuiDataType_U32, &chartOptions.maxIterations, &n1, &n5);
    ImGui::Checkbox("Use Input Mesh UVs", &chartOptions.useInputMeshUvs);
    ImGui::Checkbox("Consistent TexCoord Winding", &chartOptions.fixWinding);
    static xatlas::PackOptions packOptions;
    ImGui::Text("Pack Options");
    ImGui::SliderScalar("Max Chart Size", ImGuiDataType_U32, &packOptions.maxChartSize, &n0, &n1000);
    ImGui::SliderScalar("Padding", ImGuiDataType_U32, &packOptions.padding, &n1000, &n2);
    ImGui::SliderFloat("Texels per Unit", &packOptions.texelsPerUnit, 0, 64);
    ImGui::SliderScalar("Resolution", ImGuiDataType_U32, &packOptions.padding, &n0, &n64);
    ImGui::Checkbox("Leave space for bilinear filtering", &packOptions.bilinear);
    ImGui::Checkbox("Align charts to 4x4 blocks", &packOptions.blockAlign);
    ImGui::Checkbox("Brute Force", &packOptions.bruteForce);
    ImGui::Checkbox("Create Image", &packOptions.createImage);
    ImGui::Checkbox("Rotate Charts to Convex Hull Axis", &packOptions.rotateChartsToAxis);
    ImGui::Checkbox("Rotate Charts", &packOptions.rotateCharts);
    if (ImGui::Button("Recalculate UV Atlas and Upload to UV2s")) {
      xatlas::Generate(atlas, chartOptions, packOptions);
      for (uint32_t ix = 0; ix < atlas->meshCount; ++ix)
        copyUV2(atlas, ix, scene.renderables[ix].get().mesh);
    }
    ImGui::Text("Atlas Info");
    ImGui::Text("Size: (%d, %d)", atlas->width, atlas->height);
    ImGui::Text("# meshes: %d", atlas->meshCount);
    ImGui::Text("# atlases: %d", atlas->atlasCount);
    ImGui::Text("# charts: %d", atlas->chartCount);
    ImGui::Text("texelsPerUnit: %f", atlas->texelsPerUnit);

    for (uint32_t i = 0; i < atlas->atlasCount; i++)
      ImGui::Text("Atlas utilization: atlas[%d]: %.2f", i, atlas->utilization[i] * 100.0f);
    const ImVec2 tableOuterSize{0.f, 200.f};
    static int meshIx = 0;
    ImGui::SliderInt("Atlas Mesh Ix", &meshIx, 0, atlas->meshCount - 1);
    ImGui::Text("Obj: %s", scene.renderables[meshIx].get().name.c_str());
    if (ImGui::BeginTable("Vertices", 6, ImGuiTableFlags_Borders | ImGuiTableFlags_SizingFixedFit | ImGuiTableFlags_ScrollY, tableOuterSize)) {
      ImGui::TableSetupColumn("origIx");
      ImGui::TableSetupColumn("atlasIx");
      ImGui::TableSetupColumn("texCoord");
      ImGui::TableSetupColumn("uv");
      ImGui::TableSetupColumn("atlas");
      ImGui::TableSetupColumn("chart");
      ImGui::TableHeadersRow();
      for (size_t ix = 0; ix < atlas->meshes[meshIx].vertexCount; ++ix) {
        const xatlas::Vertex& v = atlas->meshes[meshIx].vertexArray[ix];
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
    for (auto& renderable : scene.renderables) {
      const ws::Mesh& mesh = renderable.get().mesh;
      mesh.bind();
      mesh.draw();
      mesh.unbind();
    }
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
    ws::VObjectPtr selectedObject = hierarchyWindow.draw();
    inspectorWindow.inspectObject(selectedObject);
    workshop.endFrame();
  }

  xatlas::Destroy(atlas);
  std::println("Bye!");
  return 0;
}