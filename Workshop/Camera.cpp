#include "Camera.hpp"

#include <imgui.h>

namespace ws {
const glm::vec3& ICameraView::getPosition() const {
  return position;
}

glm::mat4 ICameraView::getViewFromWorld() const {
  return glm::lookAt(position, position + getDirection(), {0, 1, 0});
}

glm::vec3 ICameraView::getForward() const {
  return getDirection();
}

glm::vec3 ICameraView::getUp() const {
  return glm::normalize(glm::cross(getRight(), getForward()));
}

glm::vec3 ICameraView::getRight() const {
  return glm::normalize(glm::cross(getDirection(), {0, 1, 0}));
}

glm::mat4 ICamera::getProjectionFromWorld() const {
  return getProjectionFromView() * getViewFromWorld();
}

glm::vec3 Camera3DView::getDirection() const {
  return {
      // cos/sin x/y/z order taken from: https://learnopengl.com/code_viewer_gh.php?code=includes/learnopengl/camera.h
      std::cos(yaw) * std::cos(pitch),
      std::sin(pitch),
      std::sin(yaw) * std::cos(pitch),
  };
}

glm::mat4 PerspectiveCameraProjection::getProjectionFromView() const {
  return glm::perspective(glm::radians(fov), aspectRatio, nearClip, farClip);
}

glm::mat4 OrthographicCameraProjection::getProjectionFromView() const {
  float left = -size * aspectRatio * 0.5f;
  float right = size * aspectRatio * 0.5f;
  float bottom = -size * 0.5f;
  float top = size * 0.5f;

  return glm::ortho(left, right, bottom, top, nearClip, farClip);
}

// -----

AutoOrbitingCamera3DViewController::AutoOrbitingCamera3DViewController(Camera3DView& cameraView)
    : cameraView(cameraView) {}

void AutoOrbitingCamera3DViewController::update(float deltaTime) {
  deltaPhi += speed * deltaTime;
  phi = phi0 + deltaPhi;
  cameraView.position = glm::vec3{
                            std::cos(phi) * std::cos(theta),
                            std::sin(theta),
                            std::sin(phi) * std::cos(theta),
                        } *
                        radius;

  cameraView.yaw = phi + std::numbers::pi_v<float>;
  cameraView.pitch = -theta;

  ImGui::Begin("Workshop");
  ImGui::Text("Orbiting Camera Controller");
  ImGui::SliderFloat("Theta", &theta, -std::numbers::pi_v<float> * 0.5, std::numbers::pi_v<float> * 0.5);
  ImGui::SliderFloat("Phi0", &phi0, -std::numbers::pi_v<float>, std::numbers::pi_v<float>);
  ImGui::SliderFloat("Speed", &speed, -2.f, 2.f);
  ImGui::SliderFloat("Radius", &radius, 0.1f, 40.f);
  ImGui::Separator();
  ImGui::End();
}

// -----

DragHelper::DragHelper(MouseButton dragButton, std::function<void()> onEnterDraggingCallback, std::function<void(const glm::vec2& drag)> onBeingDraggedCallback)
    : dragButton(dragButton), onEnterDraggingCallback(onEnterDraggingCallback), onBeingDraggedCallback(onBeingDraggedCallback) {}

void DragHelper::checkDragging(const ThreeButtonMouseState& mouseState, const glm::vec2& cursorPos) {
  if (mouseState.at(dragButton).status == MouseButtonState::Status::PRESSED) {
    // enter dragging
    if (!isBeingDragged) {
      isBeingDragged = true;
      cursor0 = cursorPos;
      onEnterDraggingCallback();  // for storing values at the beginning
      isBeingPressed = true;
    }
    // being dragged
    else {
      const glm::vec2 drag = cursorPos - cursor0;
      onBeingDraggedCallback(drag);  // for updating values while mouse is being dragged
    }
  }
  // exit dragging
  else if (isBeingPressed) {
    isBeingDragged = false;
    isBeingPressed = false;
    // onExitDraggingCallback(); should come here if every needed
  }
}

ManualCamera3DViewController::ManualCamera3DViewController(Camera3DView& cameraView)
    : cameraView(cameraView),
      rightDragHelper(
          MouseButton::LEFT,
          [&]() {
            pitch0 = cameraView.pitch;
            yaw0 = cameraView.yaw;
          },
          [&](const glm::vec2& drag) {
            cameraView.pitch = glm::clamp(pitch0 - drag.y * sensitivity, -std::numbers::pi_v<float> * 0.5f, std::numbers::pi_v<float> * 0.5f);
            cameraView.yaw = yaw0 + drag.x * sensitivity;
          }),
      middleDragHelper(
          MouseButton::RIGHT,
          [&]() {
            pos0 = cameraView.position;
          },
          [&](const glm::vec2& drag) {
            cameraView.position = pos0 + (cameraView.getRight() * drag.x - cameraView.getUp() * drag.y) * sensitivityB;
          }) {}

void ManualCamera3DViewController::update(const glm::vec2& cursorPos, const ThreeButtonMouseState& mouseState) {
  rightDragHelper.checkDragging(mouseState, cursorPos);
  middleDragHelper.checkDragging(mouseState, cursorPos);

  //float cameraSpeed = win.isKeyHeld(GLFW_KEY_LEFT_SHIFT) ? 0.1f : 1.0f;
  //if (win.isKeyHeld(GLFW_KEY_W))
  //  cameraView.position += cameraView.getForward() * cameraSpeed * deltaTime;
  //if (win.isKeyHeld(GLFW_KEY_S))
  //  cameraView.position -= cameraView.getForward() * cameraSpeed * deltaTime;
  //if (win.isKeyHeld(GLFW_KEY_A))
  //  cameraView.position -= cameraView.getRight() * cameraSpeed * deltaTime;
  //if (win.isKeyHeld(GLFW_KEY_D))
  //  cameraView.position += cameraView.getRight() * cameraSpeed * deltaTime;
  //if (win.isKeyHeld(GLFW_KEY_Q))
  //  cameraView.position += glm::vec3{0, 1, 0} * cameraSpeed * deltaTime;
  //if (win.isKeyHeld(GLFW_KEY_E))
  //  cameraView.position -= glm::vec3{0, 1, 0} * cameraSpeed * deltaTime;
}
}  // namespace ws