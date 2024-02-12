#pragma once

#include "Common.hpp"
#include "Texture.hpp"

#include <memory>
#include <optional>
#include <vector>

namespace ws {
class Framebuffer {
 public:
  // Default Framebuffer of size (width, height) with one RGBA8 color and one 32f/8 depth-stencil attachments
  Framebuffer(uint32_t width, uint32_t height, bool hasColor = true);
  Framebuffer(const std::vector<Texture::Specs>& colorSpecs, std::optional<Texture::Specs> depthStencilSpecs);
  // Default Framebuffer of size 1x1
  Framebuffer();
  Framebuffer(const Framebuffer& other) = delete;
  Framebuffer& operator=(const Framebuffer& other) = delete;
  Framebuffer(Framebuffer&& other) = default;
  Framebuffer& operator=(Framebuffer&& other) = default;
  ~Framebuffer();

  // factories
  static Framebuffer makeDefaultColorOnly(uint32_t width, uint32_t height);
  static Framebuffer makeDefaultDepthOnly(uint32_t width, uint32_t height);

  uint32_t getId() const { return id; }
  void bind() const;
  static void unbind();
  std::vector<Texture>& getColorAttachments();
  Texture& getFirstColorAttachment();
  Texture& getDepthAttachment();
  // TODO: add recreateIfNeeded(uint32_t width, uint32_t height) method, for MSAA change
  void resizeIfNeeded(uint32_t width, uint32_t height);
  static void clearColor(uint32_t id, const glm::vec4& color = {0, 0, 0, 1});
  static void clearDepth(uint32_t id, float depth = 1.0f);
  static void clearStencil(uint32_t id, int32_t stencil);
  static void clear(uint32_t id, const glm::vec4& color = {0, 0, 0, 1}, float depth = 1.0f);
  void clearColor(const glm::vec4& color = {0, 0, 0, 1}) const;
  void clearDepth(float depth = 1.0f) const;
  void clearStencil(int32_t stencil) const;
  void clear(const glm::vec4& color = {0, 0, 0, 1}, float depth = 1.0f) const;

 private:
  GlHandle id;
  uint32_t width = 0;
  uint32_t height = 0;
  std::vector<Texture> colorAttachments;
  std::optional<Texture> depthStencilAttachment;

 private:
  void attachAttachments(int32_t level = 0) const;
};
}  // namespace ws