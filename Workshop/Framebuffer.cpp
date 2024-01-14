#include "Framebuffer.hpp"

#include <glad/gl.h>

#include <cassert>
#include <cstdio>
#include <ranges>

namespace ws {
Framebuffer::Framebuffer(const std::vector<Texture::Specs>& colorSpecs, std::optional<Texture::Specs> depthStencilSpecs)
    : id([this, &colorSpecs, &depthStencilSpecs]() { 
        assert(!colorSpecs.empty() || depthStencilSpecs.has_value()); 
        uint32_t id; glGenFramebuffers(1, &id); glBindFramebuffer(GL_FRAMEBUFFER, id); return id; }()),
      width([this, &colorSpecs, &depthStencilSpecs]() { return !colorSpecs.empty() ? colorSpecs[0].width : depthStencilSpecs.value().width; }()),
      height([this, &colorSpecs, &depthStencilSpecs]() { return !colorSpecs.empty() ? colorSpecs[0].height : depthStencilSpecs.value().height; }()) {
  std::vector<uint32_t> drawBuffers;
  for (const auto& [ix, spec] : colorSpecs | std::views::enumerate) {
    assert(spec.width == width);
    assert(spec.height == height);
    colorAttachments.emplace_back(spec);
    const uint32_t attachmentNo = GL_COLOR_ATTACHMENT0 + static_cast<uint32_t>(ix);
    glFramebufferTexture2D(GL_FRAMEBUFFER, attachmentNo, GL_TEXTURE_2D, colorAttachments[ix].getId(), 0);
    drawBuffers.push_back(attachmentNo);
  }
  if (depthStencilSpecs.has_value()) {
    assert(depthStencilSpecs.value().width == width);
    assert(depthStencilSpecs.value().height == height);
    depthStencilAttachment.emplace(depthStencilSpecs.value());
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_TEXTURE_2D, depthStencilAttachment.value().getId(), 0);
  }
  if (colorAttachments.empty()) {
    glDrawBuffer(GL_NONE);
    glReadBuffer(GL_NONE);
  } else {
    glDrawBuffers(drawBuffers.size(), drawBuffers.data());
  }

  assert(glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT);
  assert(glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_INCOMPLETE_DIMENSIONS_EXT);
  assert(glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT);
  assert(glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_UNSUPPORTED);
  assert(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE);

  unbind();
}

Framebuffer::Framebuffer(uint32_t w, uint32_t h, bool hasColor)
    : Framebuffer(
        hasColor ? std::vector<Texture::Specs>{Texture::Specs{w, h, Texture::Format::RGBA8, Texture::Filter::Nearest, Texture::Wrap::Repeat}} : std::vector<Texture::Specs>{},
        Texture::Specs{w, h, Texture::Format::Depth32fStencil8, Texture::Filter::Nearest, Texture::Wrap::ClampToBorder}
      ) {}

Framebuffer::Framebuffer()
    : Framebuffer(1, 1) {}

Framebuffer::~Framebuffer() {
  // "The name zero is reserved by the GL and is silently ignored, should it occur in framebuffers, as are other unused names."
  glDeleteFramebuffers(1, &id);
}

Framebuffer Framebuffer::makeDefaultColorOnly(uint32_t width, uint32_t height) {
  const Texture::Specs defaultColorSpec{width, height, Texture::Format::RGBA8, Texture::Filter::Nearest, Texture::Wrap::Repeat};
  const std::vector<Texture::Specs> colorSpecs = {defaultColorSpec};
  return Framebuffer(colorSpecs, {});
}

Framebuffer Framebuffer::makeDefaultDepthOnly(uint32_t width, uint32_t height) {
  const Texture::Specs defaultDepthSpec{width, height, Texture::Format::Depth32fStencil8, Texture::Filter::Nearest, Texture::Wrap::ClampToBorder};
  return Framebuffer({}, {defaultDepthSpec});
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
  assert(depthStencilAttachment.has_value());

  return depthStencilAttachment.value();
}

void Framebuffer::resizeIfNeeded(uint32_t w, uint32_t h) {
  if (width == w && height == h)
    return;
  width = w;
  height = h;
  for (auto& colorAttachment : colorAttachments)
    colorAttachment.resize(width, height);
  if (depthStencilAttachment.has_value() and depthStencilAttachment.value().isValid())
    depthStencilAttachment.value().resize(width, height);
}
}  // namespace ws