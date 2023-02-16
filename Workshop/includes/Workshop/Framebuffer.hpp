#pragma once

#include "Common.hpp"
#include "Texture.hpp"

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
  Texture& getColorAttachment();
  // TODO: add recreateIfNeeded(uint32_t width, uint32_t height) method, for MSAA change
  void resizeIfNeeded(uint32_t width, uint32_t height);

 private:
  uint32_t fbo{INVALID};
  uint32_t width = 0;
  uint32_t height = 0;
  Texture texColor;
  Texture texDepthStencil;
};
}  // namespace ws