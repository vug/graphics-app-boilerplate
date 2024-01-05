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

#include <fstream>
#include <iostream>
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
  whiteTex.loadPixels(whiteTexPixels.data());
  assetManager.textures.emplace("white", std::move(whiteTex));
  ws::Shader mainShader{ws::ASSETS_FOLDER / "shaders/phong.vert", ws::ASSETS_FOLDER / "shaders/phong.frag"};
  ws::Shader unlitShader{ws::ASSETS_FOLDER / "shaders/unlit.vert", ws::ASSETS_FOLDER / "shaders/unlit.frag"};
  ws::Shader debugShader{SRC / "debug.vert", SRC / "debug.frag"};
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
        glm::vec3 worldNormal = glm::normalize(glm::transpose(glm::inverse(transform.getWorldFromObjectMatrix())) * glm::vec4(v.normal, 1));
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

  std::vector<xatlas::MeshDecl> meshDeclarations;
  // xatlas::SetProgressCallback(atlas, ProgressCallback, &stopwatch);
  for (auto& r : scene.renderables) {
    xatlas::MeshDecl& meshDecl = meshDeclarations.emplace_back(calcXAtlasMeshDecl(r.get().mesh, r.get().transform));
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
  //for (uint32_t ix = 0; ix < atlas->meshCount; ++ix) {
  //  copyUV2(atlas, ix, scene.renderables[ix].get().mesh);
  //}

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
    const glm::uvec2& winSize = workshop.getWindowSize();
    atlasFbo.resizeIfNeeded(atlas->width, atlas->height);

    workshop.imGuiDrawAppWindow();

    ImGui::Begin("Main");
    static glm::vec3 bgColor{42 / 256.0, 96 / 256.0, 87 / 256.0};
    static bool debugScene = false;
    ImGui::Checkbox("Debug Scene using debug shader", &debugScene);
    ImGui::ColorEdit3("BG Color", glm::value_ptr(bgColor));
    ImGui::Separator();
    static xatlas::ChartOptions chartOptions;
    ImGui::Text("Chart Options");
    uint32_t n0 = 0, n1 = 1, n5 = 5, n64 = 64, n1000 = 1000;
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
    ImGui::SliderScalar("Padding", ImGuiDataType_U32, &packOptions.padding, &n0, &n5);
    ImGui::SliderFloat("Texels per Unit", &packOptions.texelsPerUnit, 0, 64);
    ImGui::SliderScalar("Resolution", ImGuiDataType_U32, &packOptions.resolution, &n0, &n64);
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
    if (ImGui::Button("Save xatlas generated image") && atlas->image)
      stbi_write_png("uv_atlas_xatlas.png", atlas->width, atlas->height, 4, atlas->image, sizeof(uint32_t) * atlas->width);
    ImGui::SameLine();
    if (ImGui::Button("Save my Uv Atlas Image")) {
      const ws::Texture& tex = atlasFbo.getFirstColorAttachment();
      const uint32_t w = tex.specs.width, h = tex.specs.height;
      uint32_t* pixels = new uint32_t[w * h];
      glGetTextureSubImage(tex.getId(), 0, 0, 0, 0, w, h, 1, GL_RGBA, GL_UNSIGNED_BYTE, sizeof(uint32_t) * w * h, pixels);
      stbi_write_png("uv_atlas_vug.png", w, h, 4, pixels, sizeof(uint32_t) * w);
      delete[] pixels;
    }
    if (ImGui::Button("Export whole scene in world-space into single OBJ w/baked UV2")) {
      const char* modelFilename = "baked_scene.obj";
      std::println("Writing '{}'...", modelFilename);
      std::FILE* file;
      fopen_s(&file, modelFilename, "w");
      assert(file != nullptr);
      uint32_t firstVertex = 0;
      for (uint32_t i = 0; i < atlas->meshCount; i++) {
        const xatlas::Mesh& mesh = atlas->meshes[i];
        for (uint32_t v = 0; v < mesh.vertexCount; v++) {
          const xatlas::Vertex& vertex = mesh.vertexArray[v];
          // world position and normals are already stored in MeshDecls
          const float* vertexPosArr = (const float*)meshDeclarations[i].vertexPositionData;
          std::println(file, "v {:g} {:g} {:g}", vertexPosArr[vertex.xref * 3 + 0], vertexPosArr[vertex.xref * 3 + 1], vertexPosArr[vertex.xref * 3 + 2]);
          const float* vertexNormalArr = (const float*)meshDeclarations[i].vertexNormalData;
          std::println(file, "vn {:g} {:g} {:g}", vertexNormalArr[3 * vertex.xref + 0], vertexNormalArr[3 * vertex.xref + 1], vertexNormalArr[3 * vertex.xref + 2]);
          std::println(file, "vt {:g} {:g}", vertex.uv[0] / atlas->width, vertex.uv[1] / atlas->height);
        }
        std::println(file, "o {}", scene.renderables[i].get().name);
        std::println(file, "s off");
        for (uint32_t f = 0; f < mesh.indexCount; f += 3) {
          std::print(file, "f ");
          for (uint32_t j = 0; j < 3; j++) {
            const uint32_t index = firstVertex + mesh.indexArray[f + j] + 1;  // 1-indexed
            std::print(file, "{:d}/{:d}/{:d}{:c}", index, index, index, j == 2 ? '\n' : ' ');
          }
        }
        firstVertex += mesh.vertexCount;
      }
      std::fclose(file);    
    }
    if (ImGui::Button("Export every object in object-space into OBJs with w/baked UV2")) {
      for (uint32_t i = 0; i < atlas->meshCount; i++) {
        const ws::RenderableObject& r = scene.renderables[i];
        const ws::Mesh& wsMesh = r.mesh;
        const xatlas::Mesh& atlasMesh = atlas->meshes[i];

        const std::string modelFilename = std::format("{}_baked_uv2s.obj", r.name);
        std::FILE* file;
        fopen_s(&file, modelFilename.c_str(), "w");
        assert(file != nullptr);

        uint32_t firstVertex = 0;
        for (uint32_t vIx = 0; vIx < atlasMesh.vertexCount; vIx++) {
          const xatlas::Vertex& atlasVertex = atlasMesh.vertexArray[vIx];
          const ws::DefaultVertex& wsVertex = wsMesh.meshData.vertices[atlasVertex.xref];
          std::println(file, "v {:g} {:g} {:g}", wsVertex.position.x, wsVertex.position.y, wsVertex.position.z);
          std::println(file, "vn {:g} {:g} {:g}", wsVertex.normal.x, wsVertex.normal.y, wsVertex.normal.z);
          std::println(file, "vt {:g} {:g}", atlasVertex.uv[0] / atlas->width, atlasVertex.uv[1] / atlas->height);
        }
        std::println(file, "o {}", r.name);
        std::println(file, "s off");
        for (uint32_t f = 0; f < atlasMesh.indexCount; f += 3) {
          std::print(file, "f ");
          for (uint32_t j = 0; j < 3; j++) {
            const uint32_t index = firstVertex + atlasMesh.indexArray[f + j] + 1;  // 1-indexed
            std::print(file, "{:d}/{:d}/{:d}{:c}", index, index, index, j == 2 ? '\n' : ' ');
          }
        }
        std::fclose(file);
      }
    }
    if (ImGui::Button("Save UV2s")) {
      std::string filename = "uv2s.dat";
      std::ofstream out(filename, std::ios::binary);
      assert(out.is_open());
      uint32_t numAtlasMeshes = atlas->meshCount;
      out.write(reinterpret_cast<char*>(&numAtlasMeshes), sizeof(uint32_t));
      for (uint32_t i = 0; i < numAtlasMeshes; i++) {
        const ws::RenderableObject& r = scene.renderables[i];
        const xatlas::Mesh& atlasMesh = atlas->meshes[i];
        size_t objNameLength = r.name.length();
        out.write(reinterpret_cast<char*>(&objNameLength), sizeof(uint32_t));
        std::string objName = r.name;
        out.write(reinterpret_cast<const char*>(objName.c_str()), sizeof(char) * objNameLength);
        uint32_t numVertices = atlasMesh.vertexCount;
        out.write(reinterpret_cast<char*>(&numVertices), sizeof(uint32_t));
        for (uint32_t vIx = 0; vIx < atlasMesh.vertexCount; vIx++) {
          xatlas::Vertex& atlasVertex = atlasMesh.vertexArray[vIx];
          float u = atlasVertex.uv[0] / atlas->width;
          float v = atlasVertex.uv[1] / atlas->height;
          out.write(reinterpret_cast<char*>(&u), sizeof(float));
          out.write(reinterpret_cast<char*>(&v), sizeof(float));
        }
      }
    }
    if (ImGui::Button("Read UV2s")) {
      std::string filename = "uv2s.dat";
      std::ifstream in(filename, std::ios::binary);
      assert(in.is_open());
      uint32_t numMeshesInDat;
      in.read(reinterpret_cast<char*>(&numMeshesInDat), sizeof(uint32_t));
      std::println("numMeshes {}", numMeshesInDat);
      for (uint32_t i = 0; i < numMeshesInDat; i++) {
        uint32_t objNameLength;
        in.read(reinterpret_cast<char*>(&objNameLength), sizeof(uint32_t));
        std::println("objNameLength {}", objNameLength);
        char* objNamePtr = new char[objNameLength + 1];
        objNamePtr[objNameLength] = '\0';
        in.read(reinterpret_cast<char*>(objNamePtr), sizeof(char) * objNameLength);
        std::string objName{objNamePtr};
        std::println("objName {}", objName);
        uint32_t numVertices;
        in.read(reinterpret_cast<char*>(&numVertices), sizeof(uint32_t));
        std::println("numVertices {}", numVertices);
        const xatlas::Mesh& atlasMesh = atlas->meshes[i];
        ws::Mesh& mesh = scene.renderables[i].get().mesh;
        std::vector<ws::DefaultVertex>& vertices = mesh.meshData.vertices;
        for (uint32_t vIx = 0; vIx < numVertices; vIx++) {
          uint32_t ix = atlasMesh.vertexArray[vIx].xref;
          glm::vec2 uv2;
          in.read(reinterpret_cast<char*>(&uv2), sizeof(glm::vec2));
          vertices[ix].texCoord2 = uv2;
          //std::print("({:.3f},{:.3f}) ", uv2.x, uv2.y);
        }
        mesh.uploadData();
        std::println("");
      }
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

    textureViewer.draw();
    ws::VObjectPtr selectedObject = hierarchyWindow.draw();
    inspectorWindow.inspectObject(selectedObject);
    workshop.endFrame();
  }

  xatlas::Destroy(atlas);
  std::println("Bye!");
  return 0;
}