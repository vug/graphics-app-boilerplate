#include "AssetManager.hpp"

#include <functional>
#include <ranges>

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

bool AssetManager::doAllMaterialsHaveMatchingParametersAndUniforms() const {
  const auto materialMatches = materials 
    | std::views::values 
    | std::views::transform([](const auto& mat) { return mat.doParametersAndUniformsMatch(); });
  const bool doAllMatch = std::ranges::fold_left(materialMatches, true, std::logical_and{});
  return doAllMatch;
}
}