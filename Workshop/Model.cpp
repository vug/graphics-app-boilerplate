#include "Model.hpp"

#include <fmt/core.h>
#include <glad/gl.h>
#include <tiny_obj_loader.h>
#include <glm/geometric.hpp>

#include <array>
#include <iostream>
#include <numbers>
#include <numeric>
#include <unordered_map>

namespace ws {
DefaultMeshData makeQuad(const glm::vec2& dimensions) {
  const float halfWidth = .5f * dimensions.x;
  const float halfHeight = .5f * dimensions.y;

  const glm::vec3 normal{0, 0, -1};

  const glm::vec4 red{1, 0, 0, 1};
  const glm::vec4 green{0, 1, 0, 1};
  const glm::vec4 blue{0, 0, 1, 1};
  const glm::vec4 white{1, 1, 1, 1};

  const DefaultVertex topLeft{.position = {-halfWidth, halfHeight, 0}, .texCoord = {0, 1}, .normal = normal, .color = red};
  const DefaultVertex bottomLeft{.position = {-halfWidth, -halfHeight, 0}, .texCoord = {0, 0}, .normal = normal, .color = green};
  const DefaultVertex bottomRight{.position = {halfWidth, -halfHeight, 0}, .texCoord = {1, 0}, .normal = normal, .color = blue};
  const DefaultVertex topRight{.position = {halfWidth, halfHeight, 0}, .texCoord = {1, 1}, .normal = normal, .color = white};
  return DefaultMeshData{
      .vertices = {topLeft, bottomLeft, bottomRight, topRight},
      .indices = {
          0,
          1,
          2,
          0,
          2,
          3,
      }};
}

DefaultMeshData makeBox(const glm::vec3& dimensions) {
  const glm::vec3 halfDim = dimensions * 0.5f;
  const float width = halfDim.x, height = halfDim.y, depth = halfDim.z;

  // corners
  struct PartialVertex {
    glm::vec3 position;
    glm::vec4 color;
  };
  const PartialVertex p000 = {{-width, -height, -depth}, {0.0, 0.0, 0.0, 1.0}};
  const PartialVertex p001 = {{-width, -height, +depth}, {0.0, 0.0, 1.0, 1.0}};
  const PartialVertex p010 = {{-width, +height, -depth}, {0.0, 1.0, 0.0, 1.0}};
  const PartialVertex p011 = {{-width, +height, +depth}, {0.0, 1.0, 1.0, 1.0}};
  const PartialVertex p100 = {{+width, -height, -depth}, {1.0, 0.0, 0.0, 1.0}};
  const PartialVertex p101 = {{+width, -height, +depth}, {1.0, 0.0, 1.0, 1.0}};
  const PartialVertex p110 = {{+width, +height, -depth}, {1.0, 1.0, 0.0, 1.0}};
  const PartialVertex p111 = {{+width, +height, +depth}, {1.0, 1.0, 1.0, 1.0}};

  // normals
  const glm::vec3 nFront = {0.0f, 0.0f, 1.0f};
  const glm::vec3 nBack = -nFront;
  const glm::vec3 nLeft = {1.0f, 0.0, 0.0};
  const glm::vec3 nRight = -nLeft;
  const glm::vec3 nUp = {0.0f, 1.0f, 0.0f};
  const glm::vec3 nDown = -nUp;

  // faces (four corners in CCW, 1 normal)
  struct Face {
    std::array<PartialVertex, 4> corners;
    glm::vec3 normal;
  };
  const Face fBack = {{
                          p010,
                          p110,
                          p100,
                          p000,
                      },
                      nBack};
  const Face fFront = {{
                           p001,
                           p101,
                           p111,
                           p011,
                       },
                       nFront};
  const Face fLeft = {{
                          p110,
                          p111,
                          p101,
                          p100,
                      },
                      nLeft};
  const Face fRight = {{
                           p000,
                           p001,
                           p011,
                           p010,
                       },
                       nRight};
  const Face fUp = {{p010, p011, p111, p110}, nUp};
  const Face fDown = {{p100, p101, p001, p000}, nDown};

  DefaultMeshData meshData;
  const int faceIndices[] = {
      0,
      1,
      2,  // triangle 1 of quad
      0,
      2,
      3,
  };  // triangle 2 of quad
  const glm::vec2 uvs[] = {{0.0f, 0.0f}, {1.0f, 0.0f}, {1.0f, 1.0f}, {0.0f, 1.0f}};
  const Face faces[] = {fBack, fFront, fLeft, fRight, fUp, fDown};
  for (auto& face : faces) {
    for (int ix : faceIndices) {
      const PartialVertex& pv = face.corners[ix];
      meshData.vertices.emplace_back(pv.position, uvs[ix], face.normal, pv.color);
    }
  }

  // Since all 3 vertices at the same corner have different normals we couldn't reuse vertices with given DefaultVertex schema
  meshData.indices.resize(meshData.vertices.size());
  std::iota(meshData.indices.begin(), meshData.indices.end(), 0);
  return meshData;
}

DefaultMeshData makeTorus(float outerRadius, uint32_t outerSegments, float innerRadius, uint32_t innerSegments) {
  DefaultMeshData meshData;
  for (uint32_t i = 0; i < outerSegments; i++) {
    const float u = (float)i / (outerSegments - 1);
    const float outerAngle = 2.f * std::numbers::pi_v<float> * u;
    const glm::vec3 innerCenter = glm::vec3{cosf(outerAngle), sinf(outerAngle), 0.0f} * outerRadius;
    for (uint32_t j = 0; j < innerSegments; j++) {
      const float v = (float)j / (innerSegments - 1);
      const float innerAngle = 2.f * std::numbers::pi_v<float> * v;
      const glm::vec3 innerPos = glm::vec3{cosf(innerAngle) * cosf(outerAngle), cosf(innerAngle) * sinf(outerAngle), sinf(innerAngle)} * innerRadius;

      const glm::vec3 pos = innerCenter + innerPos;
      const glm::vec3 norm = glm::normalize(innerPos);
      const glm::vec2 uv = {u, v};
      const float pattern = static_cast<float>((i % 2) ^ (j % 2));
      const glm::vec4 col = glm::vec4{1.0, 1.0, 0.0, 1.0} * pattern + glm::vec4{0.0, 1.0, 1.0, 1.0} * (1.0f - pattern);

      meshData.vertices.emplace_back(DefaultVertex{pos, uv, norm, col});
    }
  }

  for (uint32_t i = 0; i < outerSegments; i++) {
    for (uint32_t j = 0; j < innerSegments; j++) {
      const uint32_t i1 = (i + 1) % outerSegments;
      const uint32_t j1 = (j + 1) % innerSegments;
      const uint32_t ix1 = i * innerSegments + j;
      const uint32_t ix2 = i * innerSegments + j1;
      const uint32_t ix3 = i1 * innerSegments + j;
      const uint32_t ix4 = i1 * innerSegments + j1;

      meshData.indices.insert(meshData.indices.begin(), {ix3, ix2, ix1});  // triangle-1
      meshData.indices.insert(meshData.indices.begin(), {ix2, ix3, ix4});  // triangle-2
    }
  }

  return meshData;
}

DefaultMeshData makeAxes() {
  DefaultMeshData xAxis = makeBox({1, 0.2, 0.2});
  DefaultMeshData yAxis = makeBox({0.2, 1, 0.2});
  DefaultMeshData zAxis = makeBox({0.2, 0.2, 1});
  DefaultMeshData axes;
  for (auto& v : xAxis.vertices) {
    v.color = {1, 0, 0, 1};
    v.position += glm::vec3{0.5f, 0, 0};
  }
  for (auto& v : yAxis.vertices) {
    v.color = {0, 1, 0, 1};
    v.position += glm::vec3{0, 0.5f, 0};
  }
  for (auto& v : zAxis.vertices) {
    v.color = {0, 0, 1, 1};
    v.position += glm::vec3{0, 0, 0.5f};
  }

  axes.vertices.insert(axes.vertices.end(), xAxis.vertices.begin(), xAxis.vertices.end());
  axes.indices.insert(axes.indices.end(), xAxis.indices.begin(), xAxis.indices.end());

  axes.vertices.insert(axes.vertices.end(), yAxis.vertices.begin(), yAxis.vertices.end());
  for (int ix : yAxis.indices)
    axes.indices.push_back(ix + static_cast<uint32_t>(xAxis.vertices.size()));

  axes.vertices.insert(axes.vertices.end(), zAxis.vertices.begin(), zAxis.vertices.end());
  for (int ix : zAxis.indices)
    axes.indices.push_back(ix + static_cast<uint32_t>(xAxis.vertices.size() + yAxis.vertices.size()));
  return axes;
}

struct VertexId {
  int posIx;
  int texIx;
  int nrmIx;

  bool operator==(const VertexId& other) const {
    return posIx == other.posIx && texIx == other.texIx && nrmIx == other.nrmIx;
  }

  struct HashFunc {
    std::size_t operator()(const VertexId& vi) const {
      std::size_t res = 0;
      hash_combine(res, vi.posIx);
      hash_combine(res, vi.texIx);
      hash_combine(res, vi.nrmIx);
      return res;
    }
  };
};

// OBJ file is a compressed format. Each attribute (position, texCoord, normal) stores unqiue values, e.g. only one normal value is stored if all vertices have the same normal etc.
// However, vertices sent to the GPU have combinations of the attributes. Here we stores only vertices with unique attributes in vertex data
// and triangles are stored as successive triplets of indices to the vertex data.
DefaultMeshData loadOBJ(const std::filesystem::path& filepath) {  // taken from https://github.com/tinyobjloader/tinyobjloader and modified
  tinyobj::ObjReaderConfig reader_config;
  tinyobj::ObjReader reader;
  // TODO: When errors happen return with a failure result. Can be done via optionals.
  if (!reader.ParseFromFile(filepath.string(), reader_config) && !reader.Error().empty())
    std::cerr << "TinyObjReader: " << reader.Error() << '\n';
  if (!reader.Warning().empty())
    std::cerr << "TinyObjReader: " << reader.Warning() << '\n';

  const tinyobj::attrib_t& attrib = reader.GetAttrib();
  const std::vector<tinyobj::shape_t>& shapes = reader.GetShapes();
  assert(shapes.size() > 0);
  // const std::vector<tinyobj::material_t>& materials = reader.GetMaterials(); TODO: use material info if mat file with the same name exists
  assert(attrib.vertices.size() % 3 == 0);  // Assert triangular mesh TODO: check should be on all faces

  DefaultMeshData meshData;
  uint32_t vertexIndex = 0;
  std::unordered_map<VertexId, uint32_t, VertexId::HashFunc> vertexToIndex;  // map from unique vertex attribute combinations to IndexBuffer index
  for (const auto& shape : shapes) {
    for (const auto& objIndex : shape.mesh.indices) {
      VertexId vId{objIndex.vertex_index, objIndex.texcoord_index, objIndex.normal_index};
      if (vertexToIndex.contains(vId))
        meshData.indices.push_back(vertexToIndex[vId]);
      else {
        meshData.vertices.emplace_back(
            glm::vec3{attrib.vertices[3 * vId.posIx], attrib.vertices[3 * vId.posIx + 1], attrib.vertices[3 * vId.posIx + 2]},
            objIndex.texcoord_index >= 0 ? glm::vec2{attrib.texcoords[2 * vId.texIx], attrib.texcoords[2 * vId.texIx + 1]} : glm::vec2{},
            glm::vec3{attrib.normals[3 * vId.nrmIx], attrib.normals[3 * vId.nrmIx + 1], attrib.normals[3 * vId.nrmIx + 2]},
            // when there is no color info in OBJ file tinyobjloader uses white
            glm::vec4{attrib.colors[3 * vId.posIx], attrib.colors[3 * vId.posIx + 1], attrib.colors[3 * vId.posIx + 2], 1});
        meshData.indices.push_back(vertexIndex);
        vertexToIndex.insert({vId, vertexIndex++});
      }
    }
  }

  return meshData;
}

Mesh::Mesh(const DefaultMeshData& meshData)
    : meshData(meshData) {
  // next power of two larger than size
  capacity = static_cast<size_t>(std::pow(2.0, std::ceil(std::log2(meshData.vertices.size()))));

  createBuffers();
  uploadData();
}

Mesh::Mesh(Mesh&& other)
    : meshData(std::move(other.meshData)),
      capacity(other.capacity),
      vertexArray(other.vertexArray),
      vertexBuffer(other.vertexBuffer),
      indexBuffer(other.indexBuffer) {
  other.capacity = 0;
  other.vertexArray = INVALID;
  other.vertexBuffer = INVALID;
  other.indexBuffer = INVALID;
  fmt::println("Moved Mesh with {} vertices via MoveCstor.", meshData.indices.size());
}

Mesh& Mesh::operator=(Mesh&& other) {
  meshData = std::move(other.meshData);
  capacity = other.capacity;
  vertexArray = other.vertexArray;
  vertexBuffer = other.vertexBuffer;
  indexBuffer = other.indexBuffer;
  other.capacity = 0;
  other.vertexArray = INVALID;
  other.vertexBuffer = INVALID;
  other.indexBuffer = INVALID;
  fmt::println("Moved Mesh with {} vertices via MoveAssign.", meshData.indices.size());
  return *this;
}

Mesh::~Mesh() {
  if (vertexBuffer != INVALID) {
    glDeleteBuffers(1, &vertexBuffer);
    vertexBuffer = INVALID;
  }
  if (indexBuffer != INVALID) {
    glDeleteBuffers(1, &indexBuffer);
    indexBuffer = INVALID;
  }
  if (vertexArray != INVALID) {
    glDeleteVertexArrays(1, &vertexArray);
    vertexArray = ws::INVALID;
  }
}

void Mesh::createBuffers() {
  glGenVertexArrays(1, &vertexArray);
  glGenBuffers(1, &vertexBuffer);
  glGenBuffers(1, &indexBuffer);

  allocateBuffers();

  static const std::vector<int32_t> sizes = {3, 2, 3, 4, 4};
  size_t offset = 0;
  for (size_t ix = 0; ix < sizes.size(); ++ix) {
    glVertexAttribPointer(ix, sizes[ix], GL_FLOAT, GL_FALSE, sizeof(DefaultVertex), (void*)offset);
    glEnableVertexAttribArray(ix);
    offset += sizes[ix] * sizeof(float);
  }

  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindVertexArray(0);
}

void Mesh::allocateBuffers() {
  glBindVertexArray(vertexArray);
  glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer);
  glBufferData(GL_ARRAY_BUFFER, sizeof(DefaultVertex) * capacity, nullptr, GL_DYNAMIC_DRAW);

  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indexBuffer);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(uint32_t) * capacity, nullptr, GL_DYNAMIC_DRAW);
}

void Mesh::uploadData() {
  glBindVertexArray(vertexArray);
  if (capacity < meshData.indices.size()) {
    do {
      if (capacity == 0)
        capacity = 1;
      else
        capacity *= 2;
    } while (capacity <= meshData.indices.size());

    allocateBuffers();
  }

  glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer);
  glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(DefaultVertex) * meshData.vertices.size(), meshData.vertices.data());
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indexBuffer);
  glBufferSubData(GL_ELEMENT_ARRAY_BUFFER, 0, sizeof(uint32_t) * meshData.indices.size(), meshData.indices.data());
  glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void Mesh::bind() const {
  glBindVertexArray(vertexArray);
}

void Mesh::unbind() const {
  glBindVertexArray(0);
}

void Mesh::draw() const {
  glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(meshData.indices.size()), GL_UNSIGNED_INT, 0);
  // glDrawElements(GL_POINTS, static_cast<GLsizei>(idxs.size()), GL_UNSIGNED_INT, 0);
  // glDrawElements(GL_LINES, static_cast<GLsizei>(idxs.size()), GL_UNSIGNED_INT, 0);
}
}  // namespace ws