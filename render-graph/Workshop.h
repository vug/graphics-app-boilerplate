#pragma once

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

namespace ws {
class Workshop {
 public:
  Workshop();
  ~Workshop();
  bool shouldStop();
  void beginFrame();
  void endFrame();

 private:
  GLFWwindow* window = nullptr;
};
}  // namespace ws