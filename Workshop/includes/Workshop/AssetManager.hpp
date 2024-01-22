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
  Texture white;
};

}