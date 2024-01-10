#pragma once

#include "Framebuffer.hpp"
#include "Texture.hpp"
#include "Scene.hpp"

#include <vector>

namespace ws {
class TextureViewer {
 public:
  TextureViewer(const std::vector<std::reference_wrapper<ws::Texture>>& textures);
  void draw();

 private:
  std::vector<std::reference_wrapper<ws::Texture>> textures;
  int ix{};
};

class HierarchyWindow {
 public:
  HierarchyWindow(Scene& scene);
  VObjectPtr draw();

 private:
  Scene& scene;
};

class InspectorWindow {
 public:
  void inspectObject(VObjectPtr objPtr);
};

class EditorWindow {
 public:
  EditorWindow(Scene& scene);
  void draw();

 private:
  Framebuffer fbo;
  Scene& scene;
};
}