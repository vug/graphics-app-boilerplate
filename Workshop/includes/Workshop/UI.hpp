#pragma once

#include "Texture.hpp"

#include <vector>

namespace ws {
class TextureViewer {
 public:
  TextureViewer(const std::vector<std::reference_wrapper<ws::Texture>>& textures);
  void draw();

 private:
  std::vector<std::reference_wrapper<ws::Texture>> textures;
  int ix{};

  void drawCombo();
  void drawTexture();
};
}