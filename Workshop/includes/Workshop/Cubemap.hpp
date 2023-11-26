#pragma once

#include "Common.hpp"

#include <glad/gl.h>

#include <filesystem>

namespace ws {
class Cubemap {
  using path = std::filesystem::path;

 public:
  Cubemap(const path& right, const path& left, const path& top, const path& bottom, const path& front, const path& back);
  Cubemap(const Cubemap& other) = delete;
  Cubemap& operator=(const Cubemap& other) = delete;
  Cubemap(Cubemap&& other) = default;
  Cubemap& operator=(Cubemap&& other) = default;
  ~Cubemap();

  uint32_t getId() const { return id; }
  bool isValid() const { return id != INVALID; }
  void bind() const;
  void unbind() const;

 private:
  uint32_t id = INVALID;

  void loadFace(GLenum faceLabel, const path& path);
};
}