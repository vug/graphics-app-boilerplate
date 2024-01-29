#pragma once

#include "Common.hpp"

#include <glad/gl.h>

#include <format>
#include <string>

namespace ws {
template <typename TUniformStruct>
class UniformBuffer {
 public:
  UniformBuffer(uint32_t uniformBlockBindingPoint)
      : id([]() { uint32_t uboId; glCreateBuffers(1, &uboId); return uboId; }()),
        debugName(std::format("UniformBuffer[{}]", static_cast<uint32_t>(id))) {
    glNamedBufferStorage(id, sizeof(TUniformStruct), nullptr, GL_DYNAMIC_STORAGE_BIT | GL_MAP_WRITE_BIT);
    glBindBufferBase(GL_UNIFORM_BUFFER, uniformBlockBindingPoint, id);
  }
  UniformBuffer(const UniformBuffer& other) = delete;
  UniformBuffer& operator=(const UniformBuffer& other) = delete;
  UniformBuffer(UniformBuffer&& other) = default;
  UniformBuffer& operator=(UniformBuffer&& other) = default;
  ~UniformBuffer() {
    glDeleteBuffers(1, &id);
  }

  void upload() {
    glNamedBufferSubData(id, 0, sizeof(TUniformStruct), &uniforms);
  }

  TUniformStruct& map() {
    TUniformStruct* ptr = static_cast<TUniformStruct*>(glMapNamedBuffer(id, GL_WRITE_ONLY));
    return *ptr;
  }

  void unmap() {
    glUnmapNamedBuffer(id); 
  }

  TUniformStruct uniforms{};

private:
  GlHandle id = INVALID;
  std::string debugName;
};
}