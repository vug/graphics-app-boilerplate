#pragma once

#include <glm/vec2.hpp>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <functional>
#include <unordered_map>

namespace ws {

enum class MouseButton {
  LEFT = 0,
  MIDDLE = 1,
  RIGHT = 2,
};

struct MouseButtonState {
  enum class Status {
    PRESSED = 0,
    RELEASED = 1,
  };

  Status status = Status::RELEASED;
  bool isDragging = false;
  glm::vec2 pressedPos{};
};

using MouseState = std::unordered_map<MouseButton, MouseButtonState>;

class ThreeButtonMouseState : public MouseState {
 public:
  ThreeButtonMouseState();

  MouseButtonState& left;
  MouseButtonState& middle;
  MouseButtonState& right;
};

void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
void mousePositionCallback(GLFWwindow* window, double xpos, double ypos);
void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mode);

// uses GLFW keys because I'm too lazy to write down my own enum class for all keys
bool isKeyPressed(int key);
int MouseButtonToGlfw(MouseButton button);
bool isMouseButtonPressed(MouseButton button);
glm::vec2 getMouseCursorPosition();

// A state machine that keeps track of mouse dragging input by the given mouse button.
// calls given callbacks at state changes
class DragHelper {
 private:
  MouseButton dragButton{};
  std::function<void()> onEnterDraggingCallback;
  std::function<void(const glm::vec2& drag)> onBeingDraggedCallback;
  // state
  bool isBeingDragged{};
  bool isBeingPressed{};
  glm::vec2 cursor0{};

 public:
  DragHelper(MouseButton dragButton, std::function<void()> onEnterDraggingCallback, std::function<void(const glm::vec2& drag)> onBeingDraggedCallback);
  // update function to call every frame
  void checkDragging(const ThreeButtonMouseState& mouseState, const glm::vec2& cursorPos);
};

}