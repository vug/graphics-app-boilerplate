#include "AssetManager.hpp"

namespace ws {
AssetManager::AssetManager() : 
  white(ws::Texture::Specs{1, 1, ws::Texture::Format::RGB8, ws::Texture::Filter::Nearest}),
  vaoEmpty([]() { uint32_t vao; glGenVertexArrays(1, &vao); return vao; }())
{ 
  std::vector<uint32_t> whiteTexPixels = {0xFFFFFF};
  white.uploadPixels(whiteTexPixels.data());
}

void AssetManager::drawWithEmptyVao(uint32_t numVertices) const {
  glBindVertexArray(vaoEmpty);
  glDrawArrays(GL_TRIANGLES, 0, numVertices);
  glBindVertexArray(0);
}
}