#pragma once

#include "Common.hpp"

#include <glad/gl.h>
#include <glm/gtc/type_ptr.hpp>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

//#include <cstdint>  // TODO: for uint32_t. try removing later
#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

namespace ws {
struct UniformBlockInfo {
  std::string name;
  int32_t index;
  int32_t dataSize;
  int32_t numUniforms;
  int32_t longestUniformNameLength;
};

struct UniformInfo {
  std::string name;
  uint32_t glType;
  int32_t index;
  int32_t offset;
  int32_t numItems;
  std::string typeName;
  int32_t sizeBytes;
};

// Abstraction corresponding to a Shader Program in OpenGL
// Keeps the same id throughout its lifetime
class Shader {
 public:
  // Just acquires a Shader Program Id from OpenGL context. No shaders compiled/linked. Invalid program.
  Shader();
  // Create a shader program and compile individual shaders from their source codes and link them to the program
  // If compilation fails, the shader program ends up in an invalid state
  Shader(const char* vertexShaderSource, const char* geometryShaderSource, const char* fragmentShaderSource);
  Shader(const char* vertexShaderSource, const char* fragmentShaderSource);
  Shader(const char* computeSource);
  // Create a shader program and compile shaders from files. Keep track of source files for reloads in the future
  Shader(std::filesystem::path vertexShader, std::filesystem::path geometryShader, std::filesystem::path fragmentShader);
  Shader(std::filesystem::path vertexShader, std::filesystem::path fragmentShader);
  Shader(std::filesystem::path computeShader);
  Shader(const Shader& other) = delete;
  Shader& operator=(const Shader& other) = delete;
  Shader(Shader&& other) = default;
  Shader& operator=(Shader&& other) = default;
  // Deallocate resources
  ~Shader();

  static void dispatchCompute(uint32_t x, uint32_t y, uint32_t z);

  void setInteger(const char* name, const int value) const;
  void setIntVector2(const char* name, const glm::ivec2& value) const;
  void setFloat(const char* name, const float value) const;
  void setVector2(const char* name, const glm::vec2& value) const;
  void setVector3(const char* name, const glm::vec3& value) const;
  void setVector4(const char* name, const glm::vec4& value) const;
  void setMatrix3(const char* name, const glm::mat3& value) const;
  void setMatrix4(const char* name, const glm::mat4& value) const;
  void blockBinding(const char* name, uint32_t binding) const;

  // Compiles shader sources into program.
  // Good for hard-coded shaders or recompiling generated shader code.
  // If compilation fails, keeps existing shaders if there are any.
  // If compilation succeeds, detaches existing shaders before linking new shaders to the program.
  bool compile(const char* vertexShaderSource, const char* geometryShaderSource, const char* fragmentShaderSource);
  bool compile(const char* vertexShaderSource, const char* fragmentShaderSource);
  bool compile(const char* computeShaderSource);
  // Compile shaders into program from given shader files. Update shader files.
  bool load(std::filesystem::path vertexShader, std::filesystem::path geometryShader, std::filesystem::path fragmentShader);
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

 public:
  static std::unordered_map<std::string, std::string> namedStrings;

 private:
  // detach attached shaders, if there are any
  // don't call on actively used shaders, if no new compiled shaders are going to be attached.
  void detachShaders();

 private:
  std::filesystem::path vertexShader;
  std::filesystem::path geometryShader;
  std::filesystem::path fragmentShader;
  std::filesystem::path computeShader;
  ws::GlHandle id;

public:
  static void makeNamedStringFromFile(const std::string& name, const std::filesystem::path& fp);
  // Glob shaderLibFolder (say assets/shaders/lib) and create a NamedString for each glsl file (say /lib/lights/PointLight.glsl)
  static void makeNamedStringsFromFolder(const std::filesystem::path& shaderLibFolder);
  void printAttributes() const;
  std::vector<UniformBlockInfo> getUniformBlockInfos() const;
  void printUniformBlocks() const;
  std::vector<UniformInfo> getUniformInfos() const;
  void printUniforms() const;
  void printSource() const;

  inline static std::unordered_map<GLenum, std::string> UNIFORM_AND_ATTRIBUTE_TYPES = {
    {GL_FLOAT, "float"},
    {GL_FLOAT_VEC2, "vec2"},
    {GL_FLOAT_VEC3, "vec3"},
    {GL_FLOAT_VEC4, "vec4"},
    {GL_DOUBLE, "double"},
    {GL_DOUBLE_VEC2, "dvec2"},
    {GL_DOUBLE_VEC3, "dvec3"},
    {GL_DOUBLE_VEC4, "dvec4"},
    {GL_INT, "int"},
    {GL_INT_VEC2, "ivec2"},
    {GL_INT_VEC3, "ivec3"},
    {GL_INT_VEC4, "ivec4"},
    {GL_UNSIGNED_INT, "unsigned int"},
    {GL_UNSIGNED_INT_VEC2, "uvec2"},
    {GL_UNSIGNED_INT_VEC3, "uvec3"},
    {GL_UNSIGNED_INT_VEC4, "uvec4"},
    {GL_BOOL, "bool"},
    {GL_BOOL_VEC2, "bvec2"},
    {GL_BOOL_VEC3, "bvec3"},
    {GL_BOOL_VEC4, "bvec4"},
    {GL_FLOAT_MAT2, "mat2"},
    {GL_FLOAT_MAT3, "mat3"},
    {GL_FLOAT_MAT4, "mat4"},
    {GL_FLOAT_MAT2x3, "mat2x3"},
    {GL_FLOAT_MAT2x4, "mat2x4"},
    {GL_FLOAT_MAT3x2, "mat3x2"},
    {GL_FLOAT_MAT3x4, "mat3x4"},
    {GL_FLOAT_MAT4x2, "mat4x2"},
    {GL_FLOAT_MAT4x3, "mat4x3"},
    {GL_DOUBLE_MAT2, "dmat2"},
    {GL_DOUBLE_MAT3, "dmat3"},
    {GL_DOUBLE_MAT4, "dmat4"},
    {GL_DOUBLE_MAT2x3, "dmat2x3"},
    {GL_DOUBLE_MAT2x4, "dmat2x4"},
    {GL_DOUBLE_MAT3x2, "dmat3x2"},
    {GL_DOUBLE_MAT3x4, "dmat3x4"},
    {GL_DOUBLE_MAT4x2, "dmat4x2"},
    {GL_DOUBLE_MAT4x3, "dmat4x3"},
    {GL_SAMPLER_1D, "sampler1D"},
    {GL_SAMPLER_2D, "sampler2D"},
    {GL_SAMPLER_3D, "sampler3D"},
    {GL_SAMPLER_CUBE, "samplerCube"},
    {GL_SAMPLER_1D_SHADOW, "sampler1DShadow"},
    {GL_SAMPLER_2D_SHADOW, "sampler2DShadow"},
    {GL_SAMPLER_1D_ARRAY, "sampler1DArray"},
    {GL_SAMPLER_2D_ARRAY, "sampler2DArray"},
    {GL_SAMPLER_1D_ARRAY_SHADOW, "sampler1DArrayShadow"},
    {GL_SAMPLER_2D_ARRAY_SHADOW, "sampler2DArrayShadow"},
    {GL_SAMPLER_2D_MULTISAMPLE, "sampler2DMS"},
    {GL_SAMPLER_2D_MULTISAMPLE_ARRAY, "sampler2DMSArray"},
    {GL_SAMPLER_CUBE_SHADOW, "samplerCubeShadow"},
    {GL_SAMPLER_BUFFER, "samplerBuffer"},
    {GL_SAMPLER_2D_RECT, "sampler2DRect"},
    {GL_SAMPLER_2D_RECT_SHADOW, "sampler2DRectShadow"},
    {GL_INT_SAMPLER_1D, "isampler1D"},
    {GL_INT_SAMPLER_2D, "isampler2D"},
    {GL_INT_SAMPLER_3D, "isampler3D"},
    {GL_INT_SAMPLER_CUBE, "isamplerCube"},
    {GL_INT_SAMPLER_1D_ARRAY, "isampler1DArray"},
    {GL_INT_SAMPLER_2D_ARRAY, "isampler2DArray"},
    {GL_INT_SAMPLER_2D_MULTISAMPLE, "isampler2DMS"},
    {GL_INT_SAMPLER_2D_MULTISAMPLE_ARRAY, "isampler2DMSArray"},
    {GL_INT_SAMPLER_BUFFER, "isamplerBuffer"},
    {GL_INT_SAMPLER_2D_RECT, "isampler2DRect"},
    {GL_UNSIGNED_INT_SAMPLER_1D, "usampler1D"},
    {GL_UNSIGNED_INT_SAMPLER_2D, "usampler2D"},
    {GL_UNSIGNED_INT_SAMPLER_3D, "usampler3D"},
    {GL_UNSIGNED_INT_SAMPLER_CUBE, "usamplerCube"},
    {GL_UNSIGNED_INT_SAMPLER_1D_ARRAY, "usampler2DArray"},
    {GL_UNSIGNED_INT_SAMPLER_2D_ARRAY, "usampler2DArray"},
    {GL_UNSIGNED_INT_SAMPLER_2D_MULTISAMPLE, "usampler2DMS"},
    {GL_UNSIGNED_INT_SAMPLER_2D_MULTISAMPLE_ARRAY, "usampler2DMSArray"},
    {GL_UNSIGNED_INT_SAMPLER_BUFFER, "usamplerBuffer"},
    {GL_UNSIGNED_INT_SAMPLER_2D_RECT, "usampler2DRect"},
    {GL_IMAGE_1D, "image1D"},
    {GL_IMAGE_2D, "image2D"},
    {GL_IMAGE_3D, "image3D"},
    {GL_IMAGE_2D_RECT, "image2DRect"},
    {GL_IMAGE_CUBE, "imageCube"},
    {GL_IMAGE_BUFFER, "imageBuffer"},
    {GL_IMAGE_1D_ARRAY, "image1DArray"},
    {GL_IMAGE_2D_ARRAY, "image2DArray"},
    {GL_IMAGE_2D_MULTISAMPLE, "image2DMS"},
    {GL_IMAGE_2D_MULTISAMPLE_ARRAY, "image2DMSArray"},
    {GL_INT_IMAGE_1D, "iimage1D"},
    {GL_INT_IMAGE_2D, "iimage2D"},
    {GL_INT_IMAGE_3D, "iimage3D"},
    {GL_INT_IMAGE_2D_RECT, "iimage2DRect"},
    {GL_INT_IMAGE_CUBE, "iimageCube"},
    {GL_INT_IMAGE_BUFFER, "iimageBuffer"},
    {GL_INT_IMAGE_1D_ARRAY, "iimage1DArray"},
    {GL_INT_IMAGE_2D_ARRAY, "iimage2DArray"},
    {GL_INT_IMAGE_2D_MULTISAMPLE, "iimage2DMS"},
    {GL_INT_IMAGE_2D_MULTISAMPLE_ARRAY, "iimage2DMSArray"},
    {GL_UNSIGNED_INT_IMAGE_1D, "uimage1D"},
    {GL_UNSIGNED_INT_IMAGE_2D, "uimage2D"},
    {GL_UNSIGNED_INT_IMAGE_3D, "uimage3D"},
    {GL_UNSIGNED_INT_IMAGE_2D_RECT, "uimage2DRect"},
    {GL_UNSIGNED_INT_IMAGE_CUBE, "uimageCube"},
    {GL_UNSIGNED_INT_IMAGE_BUFFER, "uimageBuffer"},
    {GL_UNSIGNED_INT_IMAGE_1D_ARRAY, "uimage1DArray"},
    {GL_UNSIGNED_INT_IMAGE_2D_ARRAY, "uimage2DArray"},
    {GL_UNSIGNED_INT_IMAGE_2D_MULTISAMPLE, "uimage2DMS"},
    {GL_UNSIGNED_INT_IMAGE_2D_MULTISAMPLE_ARRAY, "uimage2DMSArray"},
    {GL_UNSIGNED_INT_ATOMIC_COUNTER, "atomic_uint"},
  };

  inline static std::unordered_map<GLenum, size_t> UNIFORM_SIZES = {
    {GL_FLOAT, sizeof(GLfloat) * 1},
    {GL_FLOAT_VEC2, sizeof(GLfloat) * 2},
    {GL_FLOAT_VEC3, sizeof(GLfloat) * 3},
    {GL_FLOAT_VEC4, sizeof(GLfloat) * 4},
    //{GL_DOUBLE, sizeof(GLdouble) * 1}, // ?
    //{GL_DOUBLE_VEC2, sizeof(GLdouble) * 2},
    //{GL_DOUBLE_VEC3, sizeof(GLdouble) * 3},
    //{GL_DOUBLE_VEC4, sizeof(GLdouble) * 4},
    {GL_INT, sizeof(GLint) * 1},
    {GL_INT_VEC2, sizeof(GLint) * 2},
    {GL_INT_VEC3, sizeof(GLint) * 3},
    {GL_INT_VEC4, sizeof(GLint) * 4},
    {GL_UNSIGNED_INT, sizeof(GLuint) * 1},
    {GL_UNSIGNED_INT_VEC2, sizeof(GLuint) * 2},
    {GL_UNSIGNED_INT_VEC3, sizeof(GLuint) * 3},
    {GL_UNSIGNED_INT_VEC4, sizeof(GLuint) * 4},
    //{GL_BOOL, sizeof(GLfloat) * 1}, // ?
    //{GL_BOOL_VEC2, sizeof(GLfloat) * 2},
    //{GL_BOOL_VEC3, sizeof(GLfloat) * 3},
    //{GL_BOOL_VEC4, sizeof(GLfloat) * 4},
    {GL_FLOAT_MAT2, sizeof(GLfloat) * 2 * 2},
    {GL_FLOAT_MAT3, sizeof(GLfloat) * 3 * 3},
    {GL_FLOAT_MAT4, sizeof(GLfloat) * 4 * 4},
    //{GL_FLOAT_MAT2x3, sizeof(GLfloat) * 2 * 3}, // ?
    //{GL_FLOAT_MAT2x4, sizeof(GLfloat) * 2 * 4},
    //{GL_FLOAT_MAT3x2, sizeof(GLfloat) * 3 * 2},
    //{GL_FLOAT_MAT3x4, sizeof(GLfloat) * 3 * 4},
    //{GL_FLOAT_MAT4x2, sizeof(GLfloat) * 4 * 2},
    //{GL_FLOAT_MAT4x3, sizeof(GLfloat) * 4 * 3},
    //{GL_DOUBLE_MAT2, sizeof(GLdouble) * 2 * 2}, // ?
    //{GL_DOUBLE_MAT3, sizeof(GLdouble) * 3 * 3},
    //{GL_DOUBLE_MAT4, sizeof(GLdouble) * 4 * 4},
    //{GL_DOUBLE_MAT2x3, sizeof(GLdouble) * 2 * 3}, // ?
    //{GL_DOUBLE_MAT2x4, sizeof(GLdouble) * 2 * 4},
    //{GL_DOUBLE_MAT3x2, sizeof(GLdouble) * 3 * 2},
    //{GL_DOUBLE_MAT3x4, sizeof(GLdouble) * 3 * 4},
    //{GL_DOUBLE_MAT4x2, sizeof(GLdouble) * 4 * 2},
    //{GL_DOUBLE_MAT4x3, sizeof(GLdouble) * 4 * 3},
  };
};
}  // namespace ws