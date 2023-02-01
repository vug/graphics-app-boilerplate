#pragma once

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <string>

namespace ws {
class Workshop {
 public:
  Workshop(int width, int height, const std::string& name);
  ~Workshop();
  bool shouldStop();
  bool shouldBreakAtOpenGLDebugCallback() const { return shouldBreakAtOpenGLDebugCallback_; }
  void beginFrame();
  void endFrame();

 private:
  const char* glslVersion = "#version 300 es";
  const int openGLMajorVersion = 3;
  const int openGLMinorVersion = 1;
  bool shouldDebugOpenGL = true;
  bool shouldBreakAtOpenGLDebugCallback_ = true;
  bool useMSAA = false;
  GLFWwindow* window = nullptr;
};
}  // namespace ws