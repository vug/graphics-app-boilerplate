#pragma once

#include "Shader.hpp"

#include <glm/glm.hpp>

#include <cassert>
#include <string>
#include <unordered_map>
#include <variant>

namespace ws {

using ParamT = std::variant<int, float, glm::vec2, glm::vec3>;

template <typename T>
// concept ParamC = std::is_same_v<int, T> || std::is_same_v<float, T> || std::is_same_v<glm::vec2, T> || std::is_convertible_v<T, ParamT>;
concept ParamC = std::is_convertible_v<T, ParamT>;

template <class... Ts>
struct overloaded : Ts... {
  using Ts::operator()...;
};

template <class... Ts>
overloaded(Ts...) -> overloaded<Ts...>;

class Material {
 private:
  const Shader& shader;
  std::unordered_map<std::string, ParamT> parameters;

  void uploadUniform(const std::string& name, const ParamT& value);

 public:
  Material(const Shader& shader);

  // To Consider: Maybe not needed, maybe always add parameters with a provided value
  template <ParamC P>
  void addParameter(std::string name) {
    assert(!parameters.contains(name));
    parameters.emplace(name, P{});
  }

  void addParameter(std::string name, ParamT value);
  void setParameter(std::string name, ParamT value);

  void uploadParameters();
  std::string parametersToString();
};

}  // namespace ws