#include "Framebuffer.hpp"

#include <glad/gl.h>

#include <cassert>
#include <cstdio>
#include <ranges>

namespace ws {
Framebuffer::Framebuffer(uint32_t w, uint32_t h)
    : id([this]() { uint32_t id; glGenFramebuffers(1, &id); glBindFramebuffer(GL_FRAMEBUFFER, id); return id; }()),
      width(w),
      height(h),
      depthStencilAttachment{{width, height, Texture::Format::Depth32fStencil8, Texture::Filter::Nearest, Texture::Wrap::ClampToBorder}} {
  // couldn't use initialize colorAttachments member without triggering Texture copy-constructor. list-initialization was especially hard
  colorAttachments.emplace_back(Texture::Specs{width, height, Texture::Format::RGBA8, Texture::Filter::Nearest, Texture::Wrap::Repeat});

  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, colorAttachments[0].getId(), 0);
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_TEXTURE_2D, depthStencilAttachment.getId(), 0);

  assert(glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT);
  assert(glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_INCOMPLETE_DIMENSIONS_EXT);
  assert(glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT);
  assert(glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_UNSUPPORTED);
  assert(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE);

  unbind();
}

Framebuffer::Framebuffer()
    : Framebuffer(1, 1) {}

Framebuffer::~Framebuffer() {
  // "The name zero is reserved by the GL and is silently ignored, should it occur in framebuffers, as are other unused names."
  glDeleteFramebuffers(1, &id);
}

void Framebuffer::bind() const {
  glBindFramebuffer(GL_FRAMEBUFFER, id);
}

void Framebuffer::unbind() const {
  glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

std::vector<Texture>& Framebuffer::getColorAttachments() {
  return colorAttachments;
}

Texture& Framebuffer::getFirstColorAttachment() {
  assert(colorAttachments.size() >= 1);
  return colorAttachments.front();
}

Texture& Framebuffer::getDepthAttachment() {
  return depthStencilAttachment;
}

void Framebuffer::resizeIfNeeded(uint32_t w, uint32_t h) {
  if (width == w && height == h)
    return;
  width = w;
  height = h;
  for (auto& colorAttachment : colorAttachments)
    colorAttachment.resize(width, height);
  if (depthStencilAttachment.isValid())
    depthStencilAttachment.resize(width, height);
}
}  // namespace ws