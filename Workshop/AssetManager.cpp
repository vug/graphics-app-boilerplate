#include "AssetManager.hpp"

namespace ws {
AssetManager::AssetManager() 
  : white(ws::Texture::Specs{1, 1, ws::Texture::Format::RGB8, ws::Texture::Filter::Nearest}) { 
  std::vector<uint32_t> whiteTexPixels = {0xFFFFFF};
  white.uploadPixels(whiteTexPixels.data());
}
}