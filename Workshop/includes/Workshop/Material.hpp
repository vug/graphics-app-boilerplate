#pragma once

#include "Shader.hpp"

#include <glm/glm.hpp>

#include <cassert>
#include <string>
#include <unordered_map>
#include <variant>

namespace ws {

using ParamT = std::variant<int, float, glm::vec2, glm::vec3>;


class Material {
 private:

  void uploadUniform(const std::string& name, const ParamT& value) const;

 public:
  Shader& shader;
  std::unordered_map<std::string, ParamT> parameters;
  Material(Shader& shader);

  void uploadParameters() const;
  std::string parametersToString() const;
};

}  // namespace ws