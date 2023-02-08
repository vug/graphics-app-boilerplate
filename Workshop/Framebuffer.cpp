#include "Framebuffer.hpp"

#include <glad/gl.h>

#include <cassert>
#include <cstdio>

namespace ws {
Framebuffer::Framebuffer(uint32_t width, uint32_t height)
    : fbo([this]() { uint32_t id; glGenFramebuffers(1, &id); glBindFramebuffer(GL_FRAMEBUFFER, id); return id; }()),
      texColor{{width, height, Texture::Format::RGB8, Texture::Filter::Nearest, Texture::Wrap::Repeat}},
      texDepthStencil{{width, height, Texture::Format::Depth24Stencil8, Texture::Filter::Nearest, Texture::Wrap::ClampToBorder}} {
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texColor.getId(), 0);
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_TEXTURE_2D, texDepthStencil.getId(), 0);

  assert(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE);

  unbind();
}

Framebuffer::Framebuffer()
    : Framebuffer(1, 1) {}

Framebuffer::~Framebuffer() {
  glDeleteFramebuffers(1, &fbo);
}

void Framebuffer::bind() const {
  glBindFramebuffer(GL_FRAMEBUFFER, fbo);
}

void Framebuffer::unbind() const {
  glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

Texture& Framebuffer::getColorAttachment() {
  return texColor;
}
}  // namespace ws