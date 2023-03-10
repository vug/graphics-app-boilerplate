#pragma once

#include <glm/gtc/type_ptr.hpp>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

//#include <cstdint>  // TODO: for uint32_t. try removing later
#include <filesystem>
#include <vector>

namespace ws {
// Abstraction corresponding to a Shader Program in OpenGL
// Keeps the same id throughout its lifetime
class Shader {
 public:
  // Just acquires a Shader Program Id from OpenGL context. No shaders compiled/linked. Invalid program.
  Shader();
  // Create a shader program and compile shaders source codes.
  // If fails, ends up in an invalid state
  Shader(const char* vertexShaderSource, const char* fragmentShaderSource);
  // Create a shader program and compile shaders from files. Keep track of files for further reload
  Shader(std::filesystem::path vertexShader, std::filesystem::path fragmentShader);
  // Same for compute shaders
  Shader(const char* computeSource);
  Shader(std::filesystem::path computeShader);
  // Deallocate resources
  ~Shader();

  static void dispatchCompute(uint32_t x, uint32_t y, uint32_t z);

  void setInteger(const char* name, const int value) const;
  void setFloat(const char* name, const float value) const;
  void setVector2(const char* name, const glm::vec2& value) const;
  void setVector3(const char* name, const glm::vec3& value) const;
  void setMatrix3(const char* name, const glm::mat3& value) const;
  void setMatrix4(const char* name, const glm::mat4& value) const;
  void blockBinding(const char* name, uint32_t binding) const;

  // Compiles shader sources into program.
  // Good for hard-coded shaders or recompiling generated shader code.
  // If compilation fails, keeps existing shaders if there are any.
  // If compilation succeeds, detaches existing shaders before linking new shaders to the program.
  bool compile(const char* vertexShaderSource, const char* fragmentShaderSource);
  bool compile(const char* computeShaderSource);
  // Compile shaders into program from given shader files. Update shader files.
  bool load(std::filesystem::path vertexShader, std::filesystem::path fragmentShader);
  bool load(std::filesystem::path computeShader);
  // reload/recompile same shader files. Good for hot-reload.
  bool reload();

  // Getter for shader program id
  inline int32_t getId() const { return id; }
  // Whether a functioning shader program was created or not
  // i.e. shaders compiled and linked successfully, not "id != -1"
  bool isValid() const;
  // Asserts validity, then binds the shader.
  void bind() const;
  // UnBind the shader, usually not necessary.
  void unbind() const;
  // getter for ids of attached shaders to the program
  std::vector<uint32_t> getShaderIds() const;

 private:
  // detach attached shaders, if there are any
  // don't call on actively used shaders, if no new compiled shaders are going to be attached.
  void detachShaders();

 private:
  std::filesystem::path vertexShader;
  std::filesystem::path fragmentShader;
  std::filesystem::path computeShader;
  int32_t id{-1};

public:
  static void makeNamedStringFromFile(const std::string& name, const std::filesystem::path& fp);
  // Glob shaderLibFolder (say assets/shaders/lib) and create a NamedString for each glsl file (say /lib/lights/PointLight.glsl)
  static void makeNamedStringsFromFolder(const std::filesystem::path& shaderLibFolder);
};
}  // namespace ws