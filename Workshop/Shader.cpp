#include "Shader.hpp"
#include <glad/gl.h>

#include <cassert>
#include <fstream>
#include <iostream>
#include <string>

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

  Shader::Shader(const char* vertexShaderSource, const char* fragmentShaderSource)
      : id(glCreateProgram()) {
    compile(vertexShaderSource, fragmentShaderSource);
  }

  Shader::Shader(const char* computeSource)
      : id(glCreateProgram()) {
    compile(computeSource);
  }

  Shader::Shader(std::filesystem::path vertexShader, std::filesystem::path fragmentShader)
      : vertexShader(vertexShader), fragmentShader(fragmentShader), id(glCreateProgram()) {
    load(vertexShader, fragmentShader);
  }

  Shader::Shader(std::filesystem::path computeShader)
      : computeShader(computeShader), id(glCreateProgram()) {
    load(computeShader);
  }

  bool Shader::compile(const char* vertexShaderSource, const char* fragmentShaderSource) {
    int success;
    char infoLog[512];

    unsigned int vertex = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertex, 1, &vertexShaderSource, NULL);
    glCompileShader(vertex);
    glGetShaderiv(vertex, GL_COMPILE_STATUS, &success);
    if (!success) {
      glGetShaderInfoLog(vertex, 512, NULL, infoLog);
      std::cerr << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n"
                << infoLog << std::endl;
      return success;
    }

    unsigned int fragment = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragment, 1, &fragmentShaderSource, NULL);
    glCompileShader(fragment);
    glGetShaderiv(fragment, GL_COMPILE_STATUS, &success);
    if (!success) {
      glGetShaderInfoLog(fragment, 512, NULL, infoLog);
      std::cerr << "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n"
                << infoLog << std::endl;

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
      std::cerr << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n"
                << infoLog << std::endl;

      glDeleteShader(vertex);
      glDeleteShader(fragment);
      return success;
    }

    glDeleteShader(vertex);
    glDeleteShader(fragment);
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
      std::cerr << "ERROR::SHADER::COMPUTE::COMPILATION_FAILED\n"
                << infoLog << std::endl;
      return success;
    }

    if (isValid())
      detachShaders();

    glAttachShader(id, compute);
    glLinkProgram(id);
    glGetProgramiv(id, GL_LINK_STATUS, &success);
    if (!success) {
      glGetProgramInfoLog(id, 512, NULL, infoLog);
      std::cerr << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n"
                << infoLog << std::endl;

      glDeleteShader(compute);
      return success;
    }

    glDeleteShader(compute);
    return success;
  }

  bool Shader::load(std::filesystem::path vertex, std::filesystem::path fragment) {
    this->vertexShader = vertex;
    this->fragmentShader = fragment;

    if (vertex.empty() || fragment.empty()) {
      std::cerr << "shader object " << id << " is not associated with a shader file\n";
      return false;
    } else if (!std::filesystem::exists(vertex) || !std::filesystem::exists(fragment)) {
      std::cerr << "no shader file: " << vertex.string() << " or: " << fragment.string() << "\n";
      return false;
    }

    const std::string vertexCode = readFile(vertex);
    const std::string fragmentCode = readFile(fragment);

    return compile(vertexCode.c_str(), fragmentCode.c_str());
  }

  bool Shader::load(std::filesystem::path compute) {
    computeShader = compute;

    if (compute.empty()) {
      std::cerr << "shader object " << id << " is not associated with a shader file\n";
      return false;
    } else if (!std::filesystem::exists(compute)) {
      std::cerr << "no shader file: " << compute.string() << "\n";
      return false;
    }

    const std::string computeCode = readFile(compute);

    return compile(computeCode.c_str());
  }

  bool Shader::reload() {
    if (!vertexShader.empty() && !fragmentShader.empty())
      return load(vertexShader, fragmentShader);
    else if (!computeShader.empty())
      return load(computeShader);
    else {
      assert(false);  // incorrect shader code combination
      return false;
    }
  }

  Shader::~Shader() {
    if (isValid())
      detachShaders();
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
    const int maxShaders = 2;
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

  void Shader::SetScalar1f(const char* name, const float value) {
    const int location = glGetUniformLocation(id, name);
    glUniform1f(location, value);
  }

  void Shader::setVector2fv(const char* name, const float* value) {
    const int location = glGetUniformLocation(id, name);
    glUniform2fv(location, 1, value);
  }
  void Shader::setVector3fv(const char* name, const float* value) {
    const int location = glGetUniformLocation(id, name);
    glUniform3fv(location, 1, value);
  }
  void Shader::setMatrix3fv(const char* name, const float* value) {
    const int location = glGetUniformLocation(id, name);
    glUniformMatrix3fv(location, 1, GL_FALSE, value);
  }
  void Shader::setMatrix4fv(const char* name, const float* value) {
    const int location = glGetUniformLocation(id, name);
    glUniformMatrix4fv(location, 1, GL_FALSE, value);
  }
  void Shader::blockBinding(const char* name, uint32_t binding) {
    unsigned int index = glGetUniformBlockIndex(id, name);
    glUniformBlockBinding(id, index, binding);
  }
}