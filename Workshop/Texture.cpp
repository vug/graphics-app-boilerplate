#include "Texture.hpp"

#include <glad/gl.h>
#include <glm/gtc/type_ptr.hpp>
#include <stb_image.h>
#include <stb_image_write.h>

#include <cassert>

namespace ws {
Texture::GlSpecs Texture::getGlSpecs() const {
  GlSpecs gs{};

  switch (specs.format) {
    case Format::RGB8:
      gs.internalFormat = GL_RGB8;
      gs.format = GL_RGB;
      gs.type = GL_UNSIGNED_BYTE;
      break;
    case Format::RGBA8:
      gs.internalFormat = GL_RGBA8;
      gs.format = GL_RGBA;
      gs.type = GL_UNSIGNED_BYTE;
      break;
    case Format::Depth32f:
      gs.internalFormat = GL_DEPTH_COMPONENT32F;
      gs.format = GL_DEPTH_COMPONENT;
      gs.type = GL_FLOAT;
      break;
    case Format::Depth24Stencil8:
      gs.internalFormat = GL_DEPTH24_STENCIL8;
      gs.format = GL_DEPTH_STENCIL;
      gs.type = GL_UNSIGNED_INT_24_8;
      break;
    case Format::Depth32fStencil8:
      gs.internalFormat = GL_DEPTH32F_STENCIL8;
      gs.format = GL_DEPTH_STENCIL;
      gs.type = GL_FLOAT_32_UNSIGNED_INT_24_8_REV;
      break;
    case Format::R32i:
      gs.internalFormat = GL_R32I;
      gs.format = GL_RED_INTEGER;
      gs.type = GL_INT;
      break;
    case Format::R32f:
      gs.internalFormat = GL_R32F;
      gs.format = GL_RED;
      gs.type = GL_FLOAT;
      break;
    case Format::RGB32f:
      gs.internalFormat = GL_RGB32F;
      gs.format = GL_RGB;
      gs.type = GL_FLOAT;
      break;
    case Format::RGBA32f:
      gs.internalFormat = GL_RGBA32F;
      gs.format = GL_RGBA;
      gs.type = GL_FLOAT;
      break;
    default:
      assert(false);  // missing format conversion
      break;
  };

  switch (specs.filter) {
    case Filter::Nearest:
      gs.paramFilter = GL_NEAREST;
      break;
    case Filter::Linear:
      gs.paramFilter = GL_LINEAR;
      break;
    default:
      assert(false);  // missing filter conversion
  };

  switch (specs.wrap) {
    case Wrap::ClampToBorder:
      gs.paramWrap = GL_CLAMP_TO_BORDER;
      break;
    case Wrap::Repeat:
      gs.paramWrap = GL_REPEAT;
      break;
    default:
      assert(false);  // missing wrap conversion
  };

  return gs;
}

Texture::Texture()
    : Texture{Specs{}} {}

Texture::Texture(const Specs& specs)
    : specs(specs),
      id([]() { uint32_t texId; glGenTextures(1, &texId); return texId; }()),
      name{std::format("Texture[{}]", static_cast<uint32_t>(id))} {
  glBindTexture(GL_TEXTURE_2D, id);

  GlSpecs gs = getGlSpecs();
  glTexImage2D(GL_TEXTURE_2D, 0, gs.internalFormat, specs.width, specs.height, 0, gs.format, gs.type, specs.data);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, gs.paramFilter);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, gs.paramFilter);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, gs.paramWrap);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, gs.paramWrap);
  glBindTexture(GL_TEXTURE_2D, 0);
}

Texture::Texture(const std::filesystem::path& file)
    : Texture{
          [&file]() {
            assert(std::filesystem::exists(file));
            Specs specs;
            int width, height, nrChannels;
            unsigned char* data = stbi_load(file.string().c_str(), &width, &height, &nrChannels, 0);
            assert(data);
            specs.width = width;
            specs.height = height;
            specs.format = nrChannels == 4 ? Format::RGBA8 : Format::RGB8; 
            assert(nrChannels == 4 || nrChannels == 3);
            specs.wrap = Wrap::Repeat;
            specs.data = data;
            return specs;
          }()} {
}

void Texture::activateTexture(uint32_t no) {
  glActiveTexture(GL_TEXTURE0 + no);
}

void Texture::bind() const {
  glBindTexture(GL_TEXTURE_2D, id);
}

void Texture::unbind() {
  glBindTexture(GL_TEXTURE_2D, 0);
}

void Texture::bindToUnit(uint32_t unit) const {
	glBindTextureUnit(unit, id);
}

void Texture::unbindFromUnit(uint32_t unit) const {
  glBindTextureUnit(unit, 0);
}

void Texture::bindImageTexture(uint32_t textureUnit, Access access) const {
  GLenum glAccess{};
  switch (access) {
    case Access::Read:
      glAccess = GL_READ_ONLY;
      break;
    case Access::Write:
      glAccess = GL_WRITE_ONLY;
      break;
    case Access::ReadAndWrite:
      glAccess = GL_READ_WRITE;
      break;
    default:
      assert(false);  // unknown Access value
  }
  glBindImageTexture(textureUnit, id, 0, GL_FALSE, 0, glAccess, getGlSpecs().internalFormat);
}

void Texture::uploadPixels(const void* data) {
  bind();
  GlSpecs gs = getGlSpecs();
  glTexImage2D(GL_TEXTURE_2D, 0, gs.internalFormat, specs.width, specs.height, 0, gs.format, gs.type, data);
  unbind();
}

const uint32_t* Texture::downloadPixels() const {
	GlSpecs gs = getGlSpecs();
	const uint32_t w = specs.width, h = specs.height;
	// uint32_t stores a 4-channel pixel, however texture can have 1 to 4 channels. Beware, improve if needed.
	uint32_t* pixels = new uint32_t[w * h];
	glGetTextureSubImage(getId(), 0, 0, 0, 0, w, h, 1, gs.format, gs.type, sizeof(uint32_t) * w * h, pixels);
	return pixels;
}

void Texture::saveToImageFile(const std::filesystem::path& imgFile) const {
  const uint32_t* pixels = downloadPixels();
  stbi_write_png(imgFile.string().c_str(), specs.width, specs.height, 4, pixels, sizeof(uint32_t) * specs.width);
  delete[] pixels;
}

void Texture::resize(uint32_t width, uint32_t height) {
  specs.width = width;
  specs.height = height;
  GlSpecs gs = getGlSpecs();
  bind();
  glTexImage2D(GL_TEXTURE_2D, 0, gs.internalFormat, specs.width, specs.height, 0, gs.format, gs.type, specs.data);
  unbind();
}

bool Texture::resizeIfNeeded(uint32_t width, uint32_t height) {
  if (specs.width != width || specs.height != height) {
    resize(width, height);
    return true;
  }
  return false;
}

int Texture::getNumComponents() const {
  GlSpecs gs = getGlSpecs();
  switch (gs.internalFormat) {
    case GL_RED:
    case GL_R8:
    case GL_R16:
    case GL_R16F:
    case GL_R32F:
    case GL_RED_INTEGER:
      return 1;

    case GL_RG:
    case GL_RG8:
    case GL_RG16:
    case GL_RG16F:
    case GL_RG32F:
    case GL_RG_INTEGER:
      return 2;

    case GL_RGB:
    case GL_RGB8:
    case GL_RGB16:
    case GL_RGB16F:
    case GL_RGB32F:
    case GL_RGB_INTEGER:
      return 3;

    case GL_RGBA:
    case GL_RGBA8:
    case GL_RGBA16:
    case GL_RGBA16F:
    case GL_RGBA32F:
    case GL_RGBA_INTEGER:
      return 4;

    default:
      assert(false); // add missing case
      std::unreachable();
  }
}

void Texture::clear(ClearData data, int level) const {
  GlSpecs gs = getGlSpecs();
  std::visit(Overloaded{
    [&](int32_t integer) {
      assert(gs.format == GL_RED_INTEGER);
      //assert(gs.type == GL_INT);
      glClearTexImage(id, level, gs.format, GL_INT, &integer);
    },
    [&](float num) {
      assert(gs.format == GL_RED);
      glClearTexImage(id, level, gs.format, GL_FLOAT, &num);
    },
    [&](glm::vec3 rgbF) {
      assert(gs.format == GL_RGB);
      glClearTexImage(id, level, gs.format, GL_FLOAT, glm::value_ptr(rgbF));
    },
    [&](glm::vec4 rgbaF) {
      assert(gs.format == GL_RGBA);
      glClearTexImage(id, level, gs.format, GL_FLOAT, glm::value_ptr(rgbaF));
    },
    [&](glm::ivec3 rgbI) {
      assert(gs.format == GL_RGB);
      glClearTexImage(id, level, gs.format, GL_UNSIGNED_BYTE, glm::value_ptr(rgbI));
    },
    [&](glm::ivec4 rgbaI) {
      assert(gs.format == GL_RGBA);
      glClearTexImage(id, level, gs.format, GL_UNSIGNED_BYTE, glm::value_ptr(rgbaI));
    },

  }, data);
}

int32_t Texture::getNumLevels(uint32_t width, uint32_t height) {
  const float longerSide = static_cast<float>(std::max(width, height));
  const int maxLevels = 10;
  return std::min(static_cast<int32_t>(std::floor(std::log2f(longerSide))) + 1, maxLevels);
}

Texture::~Texture() {
  // glDeleteTextures silently ignores 0's and names that do not correspond to existing textures.
  glDeleteTextures(1, &id);
}
}  // namespace ws