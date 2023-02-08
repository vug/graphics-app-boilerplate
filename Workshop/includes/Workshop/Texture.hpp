#pragma once

#include "Common.hpp"

#include <glad/gl.h>
#include <filesystem>

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
    Depth32,
    Depth24Stencil8,
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
  ~Texture();

  static void activateTexture(uint32_t no = 0);

  uint32_t getId() const { return id; }
  void bind() const;
  void unbind() const;
  // should already be bound
  void bindImageTexture(uint32_t textureUnit, Access access) const;
  // not type-safe
  void loadPixels(const void* data);

  Specs specs;

 private:
  uint32_t id = INVALID;

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