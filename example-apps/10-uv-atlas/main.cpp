#include <Workshop/Assets.hpp>
#include <Workshop/Model.hpp>
#include <Workshop/Shader.hpp>
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

int main()
{
  std::println("Hi!");
  ws::Workshop workshop{1024, 768, "UV Atlas Generator"};

  ws::Mesh cube1{ws::loadOBJ(ws::ASSETS_FOLDER / "models/cube.obj")};
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
  std::println("   {} charts", atlas->chartCount);
  std::println("   {} atlases", atlas->atlasCount);
  for (uint32_t i = 0; i < atlas->atlasCount; i++)
    std::println("      {}: {:0.2f} utilization", i, atlas->utilization[i] * 100.0f);
  std::println("   ({}, {}) resolution", atlas->width, atlas->height);

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
    std::println("origIx -> atlasIx: {:2d} -> {:2d}, uv: ({:6.1f}, {:6.1f}), atlas: {}, chart: {}", v.xref, ix, v.uv[0], v.uv[1], v.atlasIndex, v.chartIndex);
  }

  xatlas::Destroy(atlas);

  const char *vertexShader = R"(
#version 300 es
#extension GL_EXT_separate_shader_objects : enable
precision mediump float;

layout (location = 0) out vec3 fragColor;

vec2 positions[3] = vec2[](vec2 (0.0, -0.5), vec2 (0.5, 0.5), vec2 (-0.5, 0.5));
vec3 colors[3] = vec3[](vec3 (1.0, 0.0, 0.0), vec3 (0.0, 1.0, 0.0), vec3 (0.0, 0.0, 1.0));
void main ()
{
	gl_Position = vec4 (positions[gl_VertexID], 0.0, 1.0);
	fragColor = colors[gl_VertexID];
}
  )";

  const char *fragmentShader = R"(
#version 300 es
#extension GL_EXT_separate_shader_objects : enable
precision mediump float;

layout (location = 0) in vec3 fragColor;
layout (location = 0) out vec4 outColor;

void main () { outColor = vec4 (fragColor, 1.0); }
  )";
  ws::Shader shader{vertexShader, fragmentShader};

  // VAO binding is needed in 4.6 was not needed in 3.1
  uint32_t vao;
  glGenVertexArrays(1, &vao);

  while (!workshop.shouldStop()) {
    workshop.beginFrame();

    workshop.imGuiDrawAppWindow();

    ImGui::Begin("Main");
    static glm::vec3 bgColor{42 / 256.0, 96 / 256.0, 87 / 256.0};
    ImGui::ColorEdit3("BG Color", glm::value_ptr(bgColor));
    ImGui::End();

    glBindVertexArray(vao);

    glClearColor(bgColor.x, bgColor.y, bgColor.z, 1);
    glClear(GL_COLOR_BUFFER_BIT);

    shader.bind();
    glDrawArrays(GL_TRIANGLES, 0, 3);
    shader.unbind();


    workshop.endFrame();
  }

  std::println("Bye!");
  return 0;
}