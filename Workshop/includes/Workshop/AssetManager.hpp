#pragma once

#include "Material.hpp"
#include "Model.hpp"
#include "Shader.hpp"
#include "Texture.hpp"

#include <unordered_map>

namespace ws {

class AssetManager {
 public:
  AssetManager();

  std::unordered_map<std::string, ws::Mesh> meshes;
  std::unordered_map<std::string, ws::Texture> textures;
  std::unordered_map<std::string, ws::Shader> shaders;
  std::unordered_map<std::string, ws::Material> materials;

  Texture white;
  void drawWithEmptyVao(uint32_t numVertices) const;

 private:
  uint32_t vaoEmpty;
};

}