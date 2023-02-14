#pragma once

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

  void setScalar1i(const char* name, const int value) const;
  void setScalar1f(const char* name, const float value) const;
  void setVector2fv(const char* name, const float* value) const;
  void setVector3fv(const char* name, const float* value) const;
  void setMatrix3fv(const char* name, const float* value) const;
  void setMatrix4fv(const char* name, const float* value) const;
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
};
}  // namespace ws