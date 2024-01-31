#include "Material.hpp"

#include <fmt/core.h>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/string_cast.hpp>

#include <iostream>
#include <sstream>
#include <string>

namespace ws {

Material::Material(Shader& shader)
    : shader(shader) {}

void Material::uploadUniform(const std::string& name, const ParamT& value) const {
  std::visit(Overloaded{
                 [&](int val) { shader.setInteger(name.c_str(), val); },
                 [&](float val) { shader.setFloat(name.c_str(), val); },
                 [&](const glm::vec2& val) { shader.setVector2(name.c_str(), val); },
                 [&](const glm::vec3& val) { shader.setVector3(name.c_str(), val); },
             },
             value);
}

void Material::uploadParameters() const {
  for (const auto& [name, value] : parameters)
    uploadUniform(name, value);
}

std::string Material::parametersToString() const {
  std::stringstream ss;
  for (const auto& [name, value] : parameters) {
    std::visit(Overloaded{
                   [&](int val) { ss << fmt::format("{} = {}\n", name, val); },
                   [&](float val) { ss << fmt::format("{} = {}\n", name, val); },
                   [&](const glm::vec2& val) { ss << fmt::format("{} = {}\n", name, glm::to_string(val)); },
                   [&](const glm::vec3& val) { ss << fmt::format("{} = {}\n", name, glm::to_string(val)); },
               },
               value);
  }
  return ss.str();
}

}