#pragma once

#include "Common.hpp"

#include <glad/gl.h>

#include <format>
#include <print>
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

  bool compareSizeWithUniformBlock(int32_t shaderId, std::string blockName) {
    int32_t blockIx = glGetUniformBlockIndex(shaderId, blockName.c_str());
    int32_t blockSize = -1;
    glGetActiveUniformBlockiv(shaderId, blockIx, GL_UNIFORM_BLOCK_DATA_SIZE, &blockSize);
    int32_t structSize = sizeof(TUniformStruct);
    bool result = blockSize == structSize;
    //std::println("block name \"{}\", uniform block data size {}, struct size {}. Match? {}", blockName, blockSize, structSize, result);
    return result;
  }

  TUniformStruct uniforms{};

private:
  GlHandle id = INVALID;
  std::string debugName;
};
}