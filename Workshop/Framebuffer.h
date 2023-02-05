#pragma once

#include "Common.h"
#include "Texture.h"

#include <memory>

namespace ws {
class Framebuffer {
 public:
  // Creates a Framebuffer of size 1x1
  Framebuffer();
  Framebuffer(uint32_t width, uint32_t height);
  ~Framebuffer();

  void bind() const;
  void unbind() const;
  // TODO: add recreateIfNeeded(uint32_t width, uint32_t height) method
  Texture& getColorAttachment();

 private:
  uint32_t fbo{INVALID};
  Texture texColor;
  Texture texDepthStencil;
};
}  // namespace ws