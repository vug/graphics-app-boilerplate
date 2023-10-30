#pragma once

#include "Input.hpp"

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
  uint32_t getFrameNo() const;
  float getFrameDurationMs() const;
  float getFrameRate() const;
  ThreeButtonMouseState mouseState{};

 private:
  const char* glslVersion = "#version 460";
  const int openGLMajorVersion = 4;
  const int openGLMinorVersion = 6;
  bool shouldDebugOpenGL = true;
  bool shouldBreakAtOpenGLDebugCallback_ = true;
  bool useMSAA = false;
  GLFWwindow* window = nullptr;
  uint32_t frameNo = 0;
  float time = 0.f;
  float frameDurationSec = 0.f;
  float frameRate = 0.f;
};
}  // namespace ws