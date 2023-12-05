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
}