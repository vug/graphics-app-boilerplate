#pragma once

#include "Common.hpp"

#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

#include <filesystem>
#include <vector>

namespace ws {
struct DefaultVertex {
  glm::vec3 position{0, 0, 0};
  glm::vec2 texCoord{0, 0};
  glm::vec2 texCoord2{0, 0};
  glm::vec3 normal{0, 0, 0};
  glm::vec4 color{1, 1, 1, 1};
  glm::vec4 custom{0, 0, 0, 0};
  // tangent
  // custom2 (glm::ivec4)
};

template <typename TVertex>
struct MeshData {
  std::vector<TVertex> vertices;
  std::vector<uint32_t> indices;
};

using DefaultMeshData = MeshData<DefaultVertex>;

DefaultMeshData makeQuad(const glm::vec2& dimensions);
DefaultMeshData makeBox(const glm::vec3& dimensions = {1, 1, 1});
DefaultMeshData makeTorus(float outerRadius, uint32_t outerSegments, float innerRadius, uint32_t innerSegments);
DefaultMeshData makeAxes();
DefaultMeshData loadOBJ(const std::filesystem::path& filepath);

template <class T>
inline void hash_combine(std::size_t& s, const T& v) {
  std::hash<T> h;
  s ^= h(v) + 0x9e3779b9 + (s << 6) + (s >> 2);
}

// https://www.khronos.org/opengl/wiki/Vertex_Specification_Best_Practices
class Mesh {
 public:
  Mesh(const DefaultMeshData& meshData);
  Mesh(const Mesh& other) = delete;
  Mesh& operator=(const Mesh& other) = delete;
  Mesh(Mesh&& other) = default;
  Mesh& operator=(Mesh&& other) = default;
  ~Mesh();

  DefaultMeshData meshData;
  size_t capacity{};

  GlHandle vertexArray;
  GlHandle vertexBuffer;
  GlHandle indexBuffer;

  // call after setting verts and idxs to upload them to GPU
  void uploadData();

  void bind() const;
  void unbind() const;
  void draw() const;

 private:
  void createBuffers();
  void allocateBuffers();
};
}  // namespace ws