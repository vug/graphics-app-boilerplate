#pragma once

#include "Common.hpp"

#include <glad/gl.h>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

#include <filesystem>
#include <variant>

namespace ws {
class Texture {
 public:
  enum class Format {
    R32i,
    R32f,
    RGB8,
    RGBA8,
    RGB16f,
    RGBA16f,
    RGB32f,
    RGBA32f,
    Depth32f,
    Depth24Stencil8,
    Depth32fStencil8,
  };

  enum class Filter {
    Nearest,
    Linear,
  };

  enum class Wrap {
    ClampToBorder,
    Repeat,
  };

  struct Specs {
    uint32_t width = 1;
    uint32_t height = 1;
    Format format = Format::RGB8;
    Filter filter = Filter::Linear;
    Wrap wrap = Wrap::ClampToBorder;
    const void* data = nullptr;
  };

  enum class Access {
    Read,
    Write,
    ReadAndWrite,
  };

  Texture();
  Texture(const Specs& specs);
  Texture(const std::filesystem::path& file);
  Texture(const Texture& other) = delete;
  Texture& operator=(const Texture& other) = delete;
  Texture(Texture&& other) = default;
  Texture& operator=(Texture&& other) = default;
  ~Texture();

  static void activateTexture(uint32_t no = 0);

  uint32_t getId() const { return id; }
  bool isValid() const { return id != INVALID; }
  void bind() const;
  void unbind() const;
  void bindToUnit(uint32_t unit) const;
  void unbindFromUnit(uint32_t unit) const;
  // should already be bound
  void bindImageTexture(uint32_t textureUnit, Access access) const;
  // not type-safe
  void uploadPixels(const void* pixels);
  const uint32_t* downloadPixels() const;
  void saveToImageFile(const std::filesystem::path& imgFile) const;
  void resize(uint32_t width, uint32_t height);
  const std::string& getName() const { return name; }
  int getNumComponents() const;
  using ClearData = std::variant<int32_t, float, glm::vec3, glm::vec4, glm::ivec3>;//, glm::ivec3, , glm::ivec4>;
  void clear(ClearData data, int level = 0) const;

  Specs specs;

 private:
  GlHandle id = INVALID;
  std::string name;

  struct GlSpecs {
    GLint internalFormat = -1;
    GLenum format = INVALID;
    GLenum type = INVALID;
    GLint paramFilter = -1;
    GLint paramWrap = -1;
  };

  GlSpecs getGlSpecs() const;
};
}  // namespace ws