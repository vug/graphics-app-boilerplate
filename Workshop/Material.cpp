#include "Material.hpp"

#include <fmt/core.h>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/string_cast.hpp>

#include <iostream>
#include <print>
#include <ranges>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_set>

namespace rng = std::ranges;
namespace vws = rng::views;

namespace ws {

std::unordered_set<std::string> Material::perMeshUniforms = {"u_WorldFromObject"};

Material::Material(Shader& shader)
    : shader(shader) 
{ }

bool Material::doParametersAndUniformsMatch() const {
  std::vector<UniformInfo> uniformInfos = shader.getUniformInfos();
  std::unordered_set<std::string> uniformNames = uniformInfos | vws::transform(&UniformInfo::name) | rng::to<std::unordered_set>();
  std::unordered_set<std::string> parameterNames = parameters | vws::keys | rng::to<std::unordered_set>();

  // Uniforms (that are not per mesh and not in a uniform block) without corresponding Material parameters. A material should provide a parameter for every uniform that needs it.
  auto uniformsWithoutParameters = uniformInfos 
    | vws::filter([&](auto& ui) { return !parameterNames.contains(ui.name) && ui.offset == -1 && !Material::perMeshUniforms.contains(ui.name); }) 
    | rng::to<std::vector>();
  for (const auto& ui : uniformsWithoutParameters)
    std::println("uniform {} has no corresponding parameter", ui.name);

  // Material parameters without corresponding uniforms in the Shader. Materials shouldn't have unnecessary parameters.
  auto parametersWithoutUniforms = parameterNames | vws::filter([&](auto& pName) { return !uniformNames.contains(pName); }) | rng::to<std::vector>();
  for (const auto& pName : parametersWithoutUniforms)
    std::println("parameter {} has no corresponding uniform", pName);

  return !uniformsWithoutParameters.empty() || !parametersWithoutUniforms.empty();
}

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