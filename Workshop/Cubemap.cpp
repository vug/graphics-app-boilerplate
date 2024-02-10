#include "Cubemap.hpp"

#include <stb_image.h>

#include <cassert>

namespace ws {
void Cubemap::loadFace(GLenum faceLabel, const path& path) {
  int width, height, nrChannels;
  unsigned char* data = stbi_load(path.string().c_str(), &width, &height, &nrChannels, 0);
  assert(nrChannels == 3);
  assert(data);
  if (faceLabel == GL_TEXTURE_CUBE_MAP_POSITIVE_X)
    glTextureStorage2D(id, 1, GL_RGB8, width, height);
  glTextureSubImage3D(id, 0, 0, 0, faceLabel - GL_TEXTURE_CUBE_MAP_POSITIVE_X, width, height, 1, GL_RGB, GL_UNSIGNED_BYTE, data);
  stbi_image_free(data);
}

Cubemap::Cubemap(const path& right, const path& left, const path& top, const path& bottom, const path& front, const path& back) {
  glCreateTextures(GL_TEXTURE_CUBE_MAP, 1, &id);

  loadFace(GL_TEXTURE_CUBE_MAP_POSITIVE_X, right);
  loadFace(GL_TEXTURE_CUBE_MAP_NEGATIVE_X, left);
  loadFace(GL_TEXTURE_CUBE_MAP_POSITIVE_Y, top);
  loadFace(GL_TEXTURE_CUBE_MAP_NEGATIVE_Y, bottom);
  loadFace(GL_TEXTURE_CUBE_MAP_POSITIVE_Z, front);
  loadFace(GL_TEXTURE_CUBE_MAP_NEGATIVE_Z, back);

  glTextureParameteri(id, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTextureParameteri(id, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTextureParameteri(id, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTextureParameteri(id, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTextureParameteri(id, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
}

Cubemap::~Cubemap() {
  glDeleteTextures(1, &id);
}

void Cubemap::bind() const {
  glBindTexture(GL_TEXTURE_CUBE_MAP, id);
}

void Cubemap::unbind() const {
  glBindTexture(GL_TEXTURE_CUBE_MAP, 0);
}

void Cubemap::bindToUnit(uint32_t unit) const {
  glBindTextureUnit(unit, id);
}
}