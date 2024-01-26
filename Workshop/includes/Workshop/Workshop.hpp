#pragma once

#include "Input.hpp"
#include "Shader.hpp"

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <glm/vec2.hpp>

#include <functional>
#include <optional>
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
  float getFrameDurationSec() const;
  float getFrameRate() const;
  bool getVSync() const;
  void setVSync(bool should);
  ThreeButtonMouseState mouseState{};
  std::optional<std::function<void(const glm::vec2&)>> onMouseMove;
  std::optional<std::function<void(const MouseButton button, const glm::vec2& pos0, const glm::vec2& pos)>> onMouseDragBegin;
  std::optional<std::function<void(const MouseButton button, const glm::vec2& pos0, const glm::vec2& pos)>> onMouseDragging;
  std::optional<std::function<void(const MouseButton button, const glm::vec2& pos0, const glm::vec2& pos)>> onMouseDragEnd;
  // Shaders inserted into this vector can be reloaded live
  std::vector<std::reference_wrapper<ws::Shader>> shadersToReload;
  void drawUI();

 private:
  const char* glslVersion = "#version 460";
  const int openGLMajorVersion = 4;
  const int openGLMinorVersion = 6;
  bool shouldDebugOpenGL = true;
  bool shouldBreakAtOpenGLDebugCallback_ = true;
  bool useMSAA = false;
  bool shouldVSync = true;
  GLFWwindow* window = nullptr;
  uint32_t frameNo = 0;
  float time = 0.f;
  float frameDurationSec = 0.f;
  float frameRate = 0.f;
};
}  // namespace ws