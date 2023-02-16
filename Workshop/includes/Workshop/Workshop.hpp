#pragma once

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <glm/vec2.hpp>

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
  inline GLFWwindow* getGLFWwindow() const { return window; };
  glm::uvec2 getWindowSize() const;

 private:
  const char* glslVersion = "#version 430";
  const int openGLMajorVersion = 4;
  const int openGLMinorVersion = 3;
  bool shouldDebugOpenGL = true;
  bool shouldBreakAtOpenGLDebugCallback_ = true;
  bool useMSAA = false;
  GLFWwindow* window = nullptr;
};
}  // namespace ws