#pragma once

#include "Shader.hpp"
#include "Texture.hpp"

#include <glm/glm.hpp>

#include <cassert>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <variant>

namespace ws {

using ParamT = std::variant<int, float, glm::vec2, glm::vec3, std::reference_wrapper<Texture>>;


class Material {
 private:
  void uploadUniform(const std::string& name, const ParamT& value) const;

 public:
  bool doParametersAndUniformsMatch() const;
  void uploadParameters() const;
  std::string parametersToString() const;

 public:
  Shader& shader;
  std::unordered_map<std::string, ParamT> parameters;

  static std::unordered_set<std::string> perMeshUniforms;
};

}  // namespace ws