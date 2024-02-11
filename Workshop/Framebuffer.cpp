#include "Framebuffer.hpp"

#include <glad/gl.h>
#include <glm/gtc/type_ptr.hpp>

#include <cassert>
#include <cstdio>
#include <ranges>

namespace ws {
Framebuffer::Framebuffer(const std::vector<Texture::Specs>& colorSpecs, std::optional<Texture::Specs> depthStencilSpecs)
    : id([this, &colorSpecs, &depthStencilSpecs]() { 
        assert(!colorSpecs.empty() || depthStencilSpecs.has_value()); 
        uint32_t id; glCreateFramebuffers(1, &id); return id; }()),
      width([this, &colorSpecs, &depthStencilSpecs]() { return !colorSpecs.empty() ? colorSpecs[0].width : depthStencilSpecs.value().width; }()),
      height([this, &colorSpecs, &depthStencilSpecs]() { return !colorSpecs.empty() ? colorSpecs[0].height : depthStencilSpecs.value().height; }()) {
  for (const auto& [ix, spec] : colorSpecs | std::views::enumerate) {
    assert(spec.width == width);
    assert(spec.height == height);
    colorAttachments.emplace_back(spec);
  }
  if (depthStencilSpecs.has_value()) {
    assert(depthStencilSpecs.value().width == width);
    assert(depthStencilSpecs.value().height == height);
    depthStencilAttachment.emplace(depthStencilSpecs.value());
  }

  attachAttachments();
}

void Framebuffer::attachAttachments(int32_t level) const {
  std::vector<uint32_t> drawBuffers;
  for (const auto& [ix, spec] : colorAttachments | std::views::enumerate) {
    const uint32_t attachmentNo = GL_COLOR_ATTACHMENT0 + static_cast<uint32_t>(ix);
    glNamedFramebufferTexture(id, attachmentNo, colorAttachments[ix].getId(), level);
    drawBuffers.push_back(attachmentNo);
  }
  if (depthStencilAttachment.has_value())
    glNamedFramebufferTexture(id, GL_DEPTH_STENCIL_ATTACHMENT, depthStencilAttachment.value().getId(), 0);

  assert(glCheckNamedFramebufferStatus(id, GL_FRAMEBUFFER) != GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT);
  assert(glCheckNamedFramebufferStatus(id, GL_FRAMEBUFFER) != GL_FRAMEBUFFER_INCOMPLETE_DIMENSIONS_EXT);
  assert(glCheckNamedFramebufferStatus(id, GL_FRAMEBUFFER) != GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT);
  assert(glCheckNamedFramebufferStatus(id, GL_FRAMEBUFFER) != GL_FRAMEBUFFER_UNSUPPORTED);
  assert(glCheckNamedFramebufferStatus(id, GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE);

  if (colorAttachments.empty()) {
    glNamedFramebufferDrawBuffer(id, GL_NONE);
    glNamedFramebufferReadBuffer(id, GL_NONE);
  } else {
    // by default we'll assume we'll write into all color attachments
    glNamedFramebufferDrawBuffers(id, static_cast<int>(drawBuffers.size()), drawBuffers.data());
  }
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

void Framebuffer::unbind() {
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

  attachAttachments();
}

void Framebuffer::clearColor(uint32_t id, const glm::vec4& color) {
  glClearNamedFramebufferfv(id, GL_COLOR, 0, glm::value_ptr(color));
}
void Framebuffer::clearDepth(uint32_t id, float depth) {
  glClearNamedFramebufferfv(id, GL_DEPTH, 0, &depth);
}
void Framebuffer::clear(uint32_t id, const glm::vec4& color, float depth) {
  clearColor(id, color);
  clearDepth(id, depth);
}
void Framebuffer::clearColor(const glm::vec4& color) const {
  clearColor(id, color);
}
void Framebuffer::clearDepth(float depth) const {
  clearDepth(id, depth);
}
void Framebuffer::clear(const glm::vec4& color, float depth) const {
  clear(id, color, depth);
}
}  // namespace ws