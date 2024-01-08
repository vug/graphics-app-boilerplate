#include "Input.hpp"
#include "Workshop.hpp"

#include <print>

namespace ws {

MouseButton glfwToMouseButton(int button) {
  switch (button) {
    case GLFW_MOUSE_BUTTON_LEFT:
      return MouseButton::LEFT;
    case GLFW_MOUSE_BUTTON_MIDDLE:
      return MouseButton::MIDDLE;
    case GLFW_MOUSE_BUTTON_RIGHT:
      return MouseButton::RIGHT;
    default:
      assert(false);  // not implemented button
      throw;
  }
}

ThreeButtonMouseState::ThreeButtonMouseState()
    : MouseState({
          {MouseButton::LEFT, MouseButtonState{}},
          {MouseButton::MIDDLE, MouseButtonState{}},
          {MouseButton::RIGHT, MouseButtonState{}},
      }),
      left(at(MouseButton::LEFT)),
      middle(at(MouseButton::MIDDLE)),
      right(at(MouseButton::RIGHT)) {}

void mouseButtonCallback(GLFWwindow* window, int button, int action, [[maybe_unused]] int mods) {
  // buttons: {GLFW_MOUSE_BUTTON_LEFT, GLFW_MOUSE_BUTTON_MIDDLE, GLFW_MOUSE_BUTTON_RIGHT}
  // actions: {GLFW_PRESS, GLFW_RELEASE}
  // modes: {GLFW_MOD_SHIFT, GLFW_MOD_CONTROL, GLFW_MOD_ALT}

  void* ptr = glfwGetWindowUserPointer(window);
  if (!ptr)
    return;
  Workshop* ws = static_cast<Workshop*>(ptr);
  const MouseButton mouseButton = glfwToMouseButton(button);
  MouseButtonState& mouseButtonState = ws->mouseState[mouseButton];

  if (action == GLFW_PRESS) {
    //std::print("PRESSED button {}, action {}\n", button, action);
    mouseButtonState.status = MouseButtonState::Status::PRESSED;
    double x, y;
    glfwGetCursorPos(window, &x, &y);
    mouseButtonState.pressedPos = {x, y};
  } else if (action == GLFW_RELEASE) {
    //std::print("RELEASED button {}, action {}\n", button, action);
    mouseButtonState.status = MouseButtonState::Status::RELEASED;
    double x, y;
    glfwGetCursorPos(window, &x, &y);
    if (mouseButtonState.isDragging) {
      if (ws->onMouseDragEnd.has_value())
        ws->onMouseDragEnd.value()(mouseButton, mouseButtonState.pressedPos, glm::vec2{x, y});
      mouseButtonState.isDragging = false;
    }
  } else {
    assert(false);  // No other action according to GLFW manual
  }
}

void mousePositionCallback(GLFWwindow* window, double xpos, double ypos) {
  void* ptr = glfwGetWindowUserPointer(window);
  if (!ptr)
    return;
  Workshop* ws = static_cast<Workshop*>(ptr);

  for (auto& [button, state] : ws->mouseState) {
    if (state.status == MouseButtonState::Status::PRESSED) {
      if (state.isDragging) {
        if (ws->onMouseDragging.has_value())
          ws->onMouseDragging.value()(button, glm::vec2{state.pressedPos.x, state.pressedPos.y}, glm::vec2{xpos, ypos});      
      } 
      else {
        if (ws->onMouseDragBegin.has_value())
          ws->onMouseDragBegin.value()(button, glm::vec2{state.pressedPos.x, state.pressedPos.y}, glm::vec2{xpos, ypos});      
        state.isDragging = true;
      }
    }
  }

  if (ws->onMouseMove.has_value())
    ws->onMouseMove.value()({xpos, ypos});
}

void keyCallback(GLFWwindow* window, int key, [[maybe_unused]] int scancode, int action, [[maybe_unused]] int mode) {
  if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
    glfwSetWindowShouldClose(window, true);
}

bool isKeyPressed(int key) {
  return glfwGetKey(glfwGetCurrentContext(), key) == GLFW_PRESS;
}

}