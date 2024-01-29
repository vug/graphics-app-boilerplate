#include "OpenGL.hpp"

#include <glad/gl.h>

#include <print>
#include <string>
#include <vector>
#include <unordered_map>

struct FeatureNames {
  GLenum feature;
  std::string name;
};

#define GL_ENUM(e) {e, #e}

std::vector<FeatureNames> GlEnums = {
  GL_ENUM(GL_BLEND),
  GL_ENUM(GL_CULL_FACE),
  GL_ENUM(GL_DEBUG_OUTPUT),
  GL_ENUM(GL_DEBUG_OUTPUT_SYNCHRONOUS),
  GL_ENUM(GL_DEPTH_TEST),
  GL_ENUM(GL_MULTISAMPLE),
  GL_ENUM(GL_STENCIL_TEST),
};

namespace ws {
void printFeatures() {
  for (auto& [e, name] : GlEnums) {
    std::println("glEnabled {}? {}", name, glIsEnabled(e));
  }
}

void printUniformBlockLimits() {
  int32_t maxVertexUniformBlocks;
  glGetIntegerv(GL_MAX_VERTEX_UNIFORM_BLOCKS, &maxVertexUniformBlocks);
  int32_t maxGeometryUniformBlocks;
  glGetIntegerv(GL_MAX_GEOMETRY_UNIFORM_BLOCKS, &maxGeometryUniformBlocks);
  int32_t maxFragmentUniformBlocks;
  glGetIntegerv(GL_MAX_FRAGMENT_UNIFORM_BLOCKS, &maxFragmentUniformBlocks);
  int32_t maxUniformBlockSize;
  glGetIntegerv(GL_MAX_UNIFORM_BLOCK_SIZE, &maxUniformBlockSize);
  std::println("GL_MAX_VERTEX_UNIFORM_BLOCKS {}, GL_MAX_GEOMETRY_UNIFORM_BLOCKS {}, GL_MAX_FRAGMENT_UNIFORM_BLOCKS {}, GL_MAX_UNIFORM_BLOCK_SIZE {} Bytes",
               maxVertexUniformBlocks, maxGeometryUniformBlocks, maxFragmentUniformBlocks, maxUniformBlockSize);
}
}