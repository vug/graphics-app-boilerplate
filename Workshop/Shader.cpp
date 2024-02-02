#include "Shader.hpp"

#include <fmt/core.h>
#include <glad/gl.h>

#include <cassert>
#include <fstream>
#include <iostream>
#include <print>
#include <ranges>
#include <regex>
#include <sstream>
#include <string>

namespace rng = std::ranges;
namespace vws = rng::views;

namespace ws {
std::string readFile(std::filesystem::path fp) {
  std::ifstream stream(fp, std::ios::in | std::ios::binary);
  const auto size = std::filesystem::file_size(fp);
  std::string content(size, '\0');
  stream.read(content.data(), size);
  return content;
}

Shader::Shader()
    : id(glCreateProgram()) {}

Shader::Shader(const char* vertexShaderSource, const char* geometryShaderSource, const char* fragmentShaderSource)
    : id(glCreateProgram()) {
  compile(vertexShaderSource, geometryShaderSource, fragmentShaderSource);
}

Shader::Shader(const char* vertexShaderSource, const char* fragmentShaderSource)
    : id(glCreateProgram()) {
  compile(vertexShaderSource, fragmentShaderSource);
}

Shader::Shader(const char* computeSource)
    : id(glCreateProgram()) {
  compile(computeSource);
}

Shader::Shader(std::filesystem::path vertexShader, std::filesystem::path geometryShader, std::filesystem::path fragmentShader)
    : vertexShader(vertexShader),
      geometryShader(geometryShader),
      fragmentShader(fragmentShader),
      id(glCreateProgram()) {
  if (!load(vertexShader, geometryShader, fragmentShader))
    std::println(std::cerr, "ERROR in {}, or {}, or {}.", vertexShader.string(), geometryShader.string(), fragmentShader.string());
}

Shader::Shader(std::filesystem::path vertexShader, std::filesystem::path fragmentShader)
    : vertexShader(vertexShader),
      fragmentShader(fragmentShader),
      id(glCreateProgram()) {
  if (!load(vertexShader, fragmentShader))
    std::println(std::cerr, "ERROR in {}, or {}.", vertexShader.string(), fragmentShader.string());
}

Shader::Shader(std::filesystem::path computeShader)
    : computeShader(computeShader), id(glCreateProgram()) {
  if(!load(computeShader))
    std::println(std::cerr, "ERROR in {}.", computeShader.string());
}

bool Shader::compile(const char* vertexShaderSource, const char* geometryShaderSource, const char* fragmentShaderSource) {
  int success;
  char infoLog[512];

  unsigned int vertex = glCreateShader(GL_VERTEX_SHADER);
  glShaderSource(vertex, 1, &vertexShaderSource, NULL);
  glCompileShaderIncludeARB(vertex, 0, NULL, NULL);
  glGetShaderiv(vertex, GL_COMPILE_STATUS, &success);
  if (!success) {
    glGetShaderInfoLog(vertex, 512, NULL, infoLog);
    std::println(std::cerr, "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n{}", infoLog);
    return success;
  }

  unsigned int geometry = glCreateShader(GL_GEOMETRY_SHADER);
  glShaderSource(geometry, 1, &geometryShaderSource, NULL);
  glCompileShaderIncludeARB(geometry, 0, NULL, NULL);
  glGetShaderiv(geometry, GL_COMPILE_STATUS, &success);
  if (!success) {
    glGetShaderInfoLog(geometry, 512, NULL, infoLog);
    std::println(std::cerr, "ERROR::SHADER::GEOMETRY::COMPILATION_FAILED\n{}", infoLog);
    glDeleteShader(vertex);
    return success;
  }

  unsigned int fragment = glCreateShader(GL_FRAGMENT_SHADER);
  glShaderSource(fragment, 1, &fragmentShaderSource, NULL);
  glCompileShaderIncludeARB(fragment, 0, NULL, NULL);
  glGetShaderiv(fragment, GL_COMPILE_STATUS, &success);
  if (!success) {
    glGetShaderInfoLog(fragment, 512, NULL, infoLog);
    std::println(std::cerr, "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n{}", infoLog);
    glDeleteShader(geometry);
    glDeleteShader(vertex);
    return success;
  }

  if (isValid())
    detachShaders();

  glAttachShader(id, vertex);
  glAttachShader(id, geometry);
  glAttachShader(id, fragment);
  glLinkProgram(id);
  glGetProgramiv(id, GL_LINK_STATUS, &success);
  if (!success) {
    glGetProgramInfoLog(id, 512, NULL, infoLog);
    std::println(std::cerr, "ERROR::SHADER::PROGRAM::LINKING_FAILED\n{}", infoLog);
    glDeleteShader(vertex);
    glDeleteShader(geometry);
    glDeleteShader(fragment);
    return success;
  }

  glDeleteShader(vertex);
  glDeleteShader(geometry);
  glDeleteShader(fragment);

  prepareUniformInfos();
  prepareUniformBlockInfos();

  return success;
}

bool Shader::compile(const char* vertexShaderSource, const char* fragmentShaderSource) {
  int success;
  char infoLog[512];

  unsigned int vertex = glCreateShader(GL_VERTEX_SHADER);
  glShaderSource(vertex, 1, &vertexShaderSource, NULL);
  glCompileShaderIncludeARB(vertex, 0, NULL, NULL);
  glGetShaderiv(vertex, GL_COMPILE_STATUS, &success);
  if (!success) {
    glGetShaderInfoLog(vertex, 512, NULL, infoLog);
    std::println(std::cerr, "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n{}", infoLog);
    return success;
  }

  unsigned int fragment = glCreateShader(GL_FRAGMENT_SHADER);
  glShaderSource(fragment, 1, &fragmentShaderSource, NULL);
  glCompileShaderIncludeARB(fragment, 0, NULL, NULL);
  glGetShaderiv(fragment, GL_COMPILE_STATUS, &success);
  if (!success) {
    glGetShaderInfoLog(fragment, 512, NULL, infoLog);
    std::println(std::cerr, "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n{}", infoLog);
    glDeleteShader(vertex);
    return success;
  }

  if (isValid())
    detachShaders();

  glAttachShader(id, vertex);
  glAttachShader(id, fragment);
  glLinkProgram(id);
  glGetProgramiv(id, GL_LINK_STATUS, &success);
  if (!success) {
    glGetProgramInfoLog(id, 512, NULL, infoLog);
    std::println(std::cerr, "ERROR::SHADER::PROGRAM::LINKING_FAILED\n{}", infoLog);
    glDeleteShader(vertex);
    glDeleteShader(fragment);
    return success;
  }

  glDeleteShader(vertex);
  glDeleteShader(fragment);

  prepareUniformInfos();
  prepareUniformBlockInfos();

  return success;
}

bool Shader::compile(const char* computeShaderSource) {
  int success;
  char infoLog[512];

  unsigned int compute = glCreateShader(GL_COMPUTE_SHADER);
  glShaderSource(compute, 1, &computeShaderSource, NULL);
  glCompileShader(compute);
  glGetShaderiv(compute, GL_COMPILE_STATUS, &success);
  if (!success) {
    glGetShaderInfoLog(compute, 512, NULL, infoLog);
    std::println(std::cerr, "ERROR::SHADER::COMPUTE::COMPILATION_FAILED\n{}", infoLog);
    return success;
  }

  if (isValid())
    detachShaders();

  glAttachShader(id, compute);
  glLinkProgram(id);
  glGetProgramiv(id, GL_LINK_STATUS, &success);
  if (!success) {
    glGetProgramInfoLog(id, 512, NULL, infoLog);
    std::println(std::cerr, "ERROR::SHADER::PROGRAM::LINKING_FAILED\n{}", infoLog);
    glDeleteShader(compute);
    return success;
  }

  glDeleteShader(compute);

  prepareUniformInfos();
  prepareUniformBlockInfos();

  return success;
}

bool Shader::load(std::filesystem::path vertex, std::filesystem::path geometry, std::filesystem::path fragment) {
  this->vertexShader = vertex;
  this->geometryShader = geometry;
  this->fragmentShader = fragment;

  if (geometry.empty() || vertex.empty() || fragment.empty()) {
    std::println(std::cerr, "shader object {} is not associated with a shader file", static_cast<uint32_t>(id));
    return false;
  } else if (!std::filesystem::exists(vertex)) {
    std::println(std::cerr, "no vertex shader file: {}", vertex.string());
    return false;
  } else if (!std::filesystem::exists(geometry)) {
    std::println(std::cerr, "no geometry shader file: {}", vertex.string());
    return false;
  } else if (!std::filesystem::exists(fragment)) {
    std::println(std::cerr, "no geometry fragment file: {}", vertex.string());
    return false;
  }

  const std::string vertexCode = readFile(vertex);
  const std::string geometryCode = readFile(geometry);
  const std::string fragmentCode = readFile(fragment);

  return compile(vertexCode.c_str(), geometryCode.c_str(), fragmentCode.c_str());
}

bool Shader::load(std::filesystem::path vertex, std::filesystem::path fragment) {
  this->vertexShader = vertex;
  this->fragmentShader = fragment;

  if (vertex.empty() || fragment.empty()) {
    std::println(std::cerr, "shader object {} is not associated with a shader file", static_cast<uint32_t>(id));
    return false;
  } else if (!std::filesystem::exists(vertex)) {
    std::println(std::cerr, "no vertex shader file: {}", vertex.string());
    return false;
  } else if (!std::filesystem::exists(fragment)) {
    std::println(std::cerr, "no geometry fragment file: {}", vertex.string());
    return false;
  }

  const std::string vertexCode = readFile(vertex);
  const std::string fragmentCode = readFile(fragment);

  return compile(vertexCode.c_str(), fragmentCode.c_str());
}

bool Shader::load(std::filesystem::path compute) {
  computeShader = compute;

  if (compute.empty()) {
    std::println(std::cerr, "shader object {} is not associated with a shader file", static_cast<uint32_t>(id));
    return false;
  } else if (!std::filesystem::exists(compute)) {
    std::println(std::cerr, "no geometry fragment file: {}", compute.string());
    return false;
  }

  const std::string computeCode = readFile(compute);

  return compile(computeCode.c_str());
}

bool Shader::reload() {
  if (!vertexShader.empty() && !geometryShader.empty() && !fragmentShader.empty())
    return load(vertexShader, geometryShader, fragmentShader);
  else if (!vertexShader.empty() && !fragmentShader.empty())
    return load(vertexShader, fragmentShader);
  else if (!computeShader.empty())
    return load(computeShader);
  else {
    assert(false);  // incorrect shader code combination
    return false;
  }
}

Shader::~Shader() {
  // No need to detach shaders glDeleteProgram will detach them. But also we delete shaders after compilation is completed. 
  glDeleteProgram(id);
}

bool Shader::isValid() const {
  int valid{};
  glGetProgramiv(id, GL_LINK_STATUS, &valid);
  return valid;
}

void Shader::bind() const {
  assert(isValid());
  glUseProgram(id);
}

void Shader::unbind() const {
  glUseProgram(0);
}

std::vector<uint32_t> Shader::getShaderIds() const {
  const int maxShaders = 3;
  std::vector<uint32_t> shaderIds(maxShaders);

  int count{};
  glGetAttachedShaders(id, 2, &count, shaderIds.data());
  shaderIds.resize(count);

  return shaderIds;
}

void Shader::detachShaders() {
  auto shaderIds = getShaderIds();
  for (auto shaderId : shaderIds)
    glDetachShader(id, shaderId);
}

void Shader::dispatchCompute(uint32_t x, uint32_t y, uint32_t z) {
  glDispatchCompute(x, y, z);
}

void Shader::setInteger(const char* name, const int value) const {
  const int location = glGetUniformLocation(id, name);
  glUniform1i(location, value);
}

void Shader::setIntVector2(const char* name, const glm::ivec2& value) const {
  const int location = glGetUniformLocation(id, name);
  glUniform2iv(location, 1, glm::value_ptr(value));
}

void Shader::setFloat(const char* name, const float value) const {
  const int location = glGetUniformLocation(id, name);
  glUniform1f(location, value);
}

void Shader::setVector2(const char* name, const glm::vec2& value) const {
  const int location = glGetUniformLocation(id, name);
  glUniform2fv(location, 1, glm::value_ptr(value));
}

void Shader::setVector3(const char* name, const glm::vec3& value) const {
  const int location = glGetUniformLocation(id, name);
  glUniform3fv(location, 1, glm::value_ptr(value));
}

void Shader::setVector4(const char* name, const glm::vec4& value) const {
  const int location = glGetUniformLocation(id, name);
  glUniform4fv(location, 1, glm::value_ptr(value));
}

void Shader::setMatrix3(const char* name, const glm::mat3& value) const {
  const int location = glGetUniformLocation(id, name);
  glUniformMatrix3fv(location, 1, GL_FALSE, glm::value_ptr(value));
}

void Shader::setMatrix4(const char* name, const glm::mat4& value) const {
  const int location = glGetUniformLocation(id, name);
  glUniformMatrix4fv(location, 1, GL_FALSE, glm::value_ptr(value));
}

void Shader::blockBinding(const char* name, uint32_t binding) const {
  unsigned int index = glGetUniformBlockIndex(id, name);
  glUniformBlockBinding(id, index, binding);
}

std::unordered_map<std::string, std::string> Shader::namedStrings;

void Shader::makeNamedStringFromFile(const std::string& name, const std::filesystem::path& fp) {
  assert(std::filesystem::exists(fp));
  std::ifstream file{fp};
  std::ostringstream ss;
  ss << file.rdbuf();
  std::string content = ss.str();
  Shader::namedStrings.emplace(name, content);
  glNamedStringARB(GL_SHADER_INCLUDE_ARB,
                   static_cast<int32_t>(name.length()), name.data(),
                   static_cast<int32_t>(content.length()), content.data());
}

void Shader::makeNamedStringsFromFolder(const std::filesystem::path& shaderLibFolder) {
  for (const std::filesystem::directory_entry& dirEntry : std::filesystem::recursive_directory_iterator(shaderLibFolder)) {
    if (!dirEntry.is_regular_file())
      continue;
    if (dirEntry.path().extension().string() != ".glsl")
      continue;
    const auto& shaderFullPath = dirEntry.path();
    const auto shaderRelPath = std::filesystem::relative(shaderFullPath, shaderLibFolder);
    // On Windows shaderRelPath has \ as separator instead of /
    // However glNamedStringARB requires / (?) therefore doing string manipulation for replacement
    std::string shaderRelPathStr = shaderRelPath.string();
    std::replace(shaderRelPathStr.begin(), shaderRelPathStr.end(), '\\', '/'); 
    std::string namedString{fmt::format("/lib/{}", shaderRelPathStr)};
    Shader::makeNamedStringFromFile(namedString, shaderFullPath);
  }
}

void Shader::printAttributes() const {
  int32_t longestNameLength{};
  glGetProgramiv(id, GL_ACTIVE_ATTRIBUTE_MAX_LENGTH, &longestNameLength);
  std::string name(longestNameLength, '\0');
  int32_t numActive;
  glGetProgramiv(id, GL_ACTIVE_ATTRIBUTES, &numActive);
  for (int i = 0; i < numActive; ++i) {
    int32_t nameLength;
    int32_t size;
    uint32_t type;
    glGetActiveAttrib(id, i, longestNameLength, &nameLength, &size, &type, name.data());
    name.resize(nameLength);
    std::println("attribute {} {}[{}] {}", i, name, size, ws::Shader::UNIFORM_AND_ATTRIBUTE_TYPES[type]);
  }
}

void Shader::prepareUniformInfos() {
  uniformInfos.clear();

  int32_t longestNameLength{};
  glGetProgramiv(id, GL_ACTIVE_UNIFORM_MAX_LENGTH, &longestNameLength);
  int32_t numActive;
  glGetProgramiv(id, GL_ACTIVE_UNIFORMS, &numActive);
  std::vector<int32_t> uniformIndices(numActive);
  for (int32_t i = 0; i < numActive; ++i)
    uniformIndices[i] = i;
  std::vector<int32_t> uniformOffsets(numActive);
  glGetActiveUniformsiv(id, numActive, (uint32_t*)uniformIndices.data(), GL_UNIFORM_OFFSET, uniformOffsets.data());

  for (int ix = 0; ix < numActive; ++ix) {
    std::string name(longestNameLength, '\0');
    int32_t nameLength;
    int32_t numItems;
    uint32_t type;
    glGetActiveUniform(id, ix, longestNameLength, &nameLength, &numItems, &type, name.data());
    name.resize(nameLength);
    uniformInfos.emplace_back(name, type, ix, uniformOffsets[ix], numItems, ws::Shader::UNIFORM_AND_ATTRIBUTE_TYPES[type], static_cast<int32_t>(ws::Shader::UNIFORM_SIZES[type]));
  }
}

const std::vector<UniformInfo>& Shader::getUniformInfos() const {
  return uniformInfos;
}

void Shader::printUniforms() const {
  const std::vector<UniformInfo>& uniformInfos = getUniformInfos();
  std::println("Offset Size Index Type Name NumItems");
  for (const auto& ui : uniformInfos)
    std::println("{:4d} {:4d} [{:3d}] {:10s} {} {}", ui.offset, ui.sizeBytes, ui.index, ui.typeName, ui.name, ui.numItems);
}

void Shader::prepareUniformBlockInfos() {
  uniformBlockInfos.clear();

  int32_t longestBlockNameLength{};
  glGetProgramiv(id, GL_ACTIVE_UNIFORM_BLOCK_MAX_NAME_LENGTH, &longestBlockNameLength);
  int32_t longestUniformNameLength{};
  glGetProgramiv(id, GL_ACTIVE_UNIFORM_MAX_LENGTH, &longestUniformNameLength);
  int32_t numBlocks;
  glGetProgramiv(id, GL_ACTIVE_UNIFORM_BLOCKS, &numBlocks);

  for (int ix = 0; ix < numBlocks; ++ix) {
    std::string blockName(longestBlockNameLength, '\0');
    int32_t blockNameLength;
    glGetActiveUniformBlockName(id, ix, longestBlockNameLength, &blockNameLength, blockName.data());
    blockName.resize(blockNameLength);
    int32_t blockDataSize = -1;
    glGetActiveUniformBlockiv(id, ix, GL_UNIFORM_BLOCK_DATA_SIZE, &blockDataSize);
    int32_t numUniforms = -1;
    glGetActiveUniformBlockiv(id, ix, GL_UNIFORM_BLOCK_ACTIVE_UNIFORMS, &numUniforms);

    uniformBlockInfos.emplace_back(blockName, ix, blockDataSize, numUniforms, longestUniformNameLength);
  }
}

const std::vector<UniformBlockInfo>& Shader::getUniformBlockInfos() const {
  return uniformBlockInfos;
}

void Shader::printUniformBlocks() const {
  const std::vector<UniformBlockInfo>& uniformBlockInfos = getUniformBlockInfos();
  const std::vector<UniformInfo>& allUniformInfos = getUniformInfos();

  for (const auto& [ix, ubi] : uniformBlockInfos | vws::enumerate) {
    std::vector<int32_t> uniformIndices(ubi.numUniforms);
    glGetActiveUniformBlockiv(id, ix, GL_UNIFORM_BLOCK_ACTIVE_UNIFORM_INDICES, uniformIndices.data());
    auto blockUniformInfos = uniformIndices | vws::transform([&](int32_t ix) { return allUniformInfos[ix]; }) | rng::to<std::vector>();
    rng::sort(blockUniformInfos, {}, &UniformInfo::offset);

    std::println("Uniform Block ix {}, name {} data size {} num active uniforms {}", ubi.index, ubi.name, ubi.dataSize, ubi.numUniforms);
    std::println("Offset Size Index Type Name NumItems");
    for (const auto& ui : blockUniformInfos)
      std::println("{:4d} {:4d} [{:3d}] {:10s} {} {}", ui.offset, ui.sizeBytes, ui.index, ui.typeName, ui.name, ui.numItems);

    // Check whether padding was done correctly
    for (const auto& [curr, next] : vws::adjacent<2>(blockUniformInfos))
      assert(next.offset == curr.offset + curr.sizeBytes);
    assert(ubi.dataSize == blockUniformInfos.back().offset + blockUniformInfos.back().sizeBytes);
  }
}

void Shader::printSource() const {
  static std::regex patternName("\"(.*?)\"");

  for (uint32_t sId : getShaderIds()) {
    GLint sourceLength;
    glGetShaderiv(sId, GL_SHADER_SOURCE_LENGTH, &sourceLength);
    std::string sourceCode(sourceLength, '\0');
    glGetShaderSource(sId, sourceLength, nullptr, sourceCode.data());
    std::println("shader[{}] source:", sId);

    while (sourceCode.contains("#include")) {
      size_t pos1 = sourceCode.find("#include");
      size_t pos2 = sourceCode.find("\n", pos1);
      std::string includeLine = sourceCode.substr(pos1, pos2 - pos1);
      std::smatch matches;
      bool hasFound = std::regex_search(includeLine, matches, patternName);
      assert(hasFound);
      std::string name = matches[1].str();
      std::string replacement = std::format("// INCLUDED '{}'\n{}", name, Shader::namedStrings[name]);
      sourceCode.replace(pos1, pos2 - pos1, replacement);
    }

    std::println("{}\n// END OF SHADER\n\n\n\n\n\n\n\n\n", sourceCode);
  }
}

}  // namespace ws