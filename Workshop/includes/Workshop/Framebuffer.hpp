#pragma once

#include "Common.hpp"
#include "Texture.hpp"

#include <memory>
#include <vector>

namespace ws {
class Framebuffer {
 public:
  // Default Framebuffer of size (width, height) with one RGBA8 color and one 32f/8 depth-stencil attachments
  Framebuffer(uint32_t width, uint32_t height);
  // Default Framebuffer of size 1x1
  Framebuffer();
  ~Framebuffer();

  void bind() const;
  void unbind() const;
  std::vector<Texture>& getColorAttachments();
  Texture& getFirstColorAttachment();
  // TODO: add recreateIfNeeded(uint32_t width, uint32_t height) method, for MSAA change
  void resizeIfNeeded(uint32_t width, uint32_t height);

 private:
  uint32_t fbo{INVALID};
  uint32_t width = 0;
  uint32_t height = 0;
  std::vector<Texture> colorAttachments;
  Texture depthStencilAttachment;
};
}  // namespace ws