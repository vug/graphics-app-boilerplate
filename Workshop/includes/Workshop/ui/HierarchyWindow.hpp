#pragma once

#include "../Scene.hpp"

namespace ws {
class HierarchyWindow {
 public:
  HierarchyWindow(Scene& scene);
  VObjectPtr draw(VObjectPtr clickedObject);

 private:
  Scene& scene;
};
}