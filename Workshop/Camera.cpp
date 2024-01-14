#include "Camera.hpp"

#include <glm/glm.hpp>
#include <glm/gtx/rotate_vector.hpp>
#include <imgui.h>

namespace ws {

glm::vec3 Camera::getDirection() const {
	return glm::normalize(target - position);
}

glm::mat4 Camera::getViewFromWorld() const {
	return glm::lookAt(position, position + getDirection(), {0, 1, 0});
}

glm::vec3 Camera::getForward() const {
	return getDirection();
}

glm::vec3 Camera::getUp() const {
	return glm::normalize(glm::cross(getRight(), getForward()));
}

glm::vec3 Camera::getRight() const {
	return glm::normalize(glm::cross(getDirection(), {0, 1, 0}));
}

glm::mat4 Camera::getProjectionFromView() const {
	return glm::perspective(glm::radians(fov), aspectRatio, nearClip, farClip);
}

glm::mat4 Camera::getProjectionFromWorld() const {
	return getProjectionFromView() * getViewFromWorld();
}

float Camera::getPitch() const {
	glm::vec3 dir = getDirection();
	return glm::acos(dir.z);
}

//

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

AutoOrbitingCameraController::AutoOrbitingCameraController(Camera& cam)
    : camera(cam) {}

void AutoOrbitingCameraController::update(float deltaTime) {
  deltaPhi += speed * deltaTime;
  phi = phi0 + deltaPhi;
  camera.position = camera.target + glm::vec3{
                                        std::cos(phi) * std::cos(theta),
                                        std::sin(theta),
                                        std::sin(phi) * std::cos(theta),
                                    } * radius;

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

ManualCameraController::ManualCameraController(Camera& cam)
    : camera(cam),
      leftDragHelper(
          MouseButton::LEFT,
          [&]() {
            cam0 = camera;
          },
          [&](const glm::vec2& drag) {
            const glm::vec3 oldTargetToOldPos = cam0.position - cam0.target;
            const float deltaPitch = -drag.y * sensitivity;
            glm::vec3 oldTargetToNewPos = glm::rotate(oldTargetToOldPos, deltaPitch, cam0.getRight());
            const float deltaYaw = -drag.x * sensitivity;
            oldTargetToNewPos = glm::rotate(oldTargetToNewPos, deltaYaw, glm::vec3{0, 1, 0});
            cam.position = cam0.target + oldTargetToNewPos;
          }),
      middleDragHelper(
          MouseButton::MIDDLE,
          [&]() {
            cam0 = camera;
          },
          [&](const glm::vec2& drag) {
            camera.position = cam0.position + (camera.getRight() * drag.x - camera.getUp() * drag.y) * sensitivityB;
          }), 
      rightDragHelper(
          MouseButton::RIGHT,
          [&]() {
            cam0 = camera;
          },
          [&](const glm::vec2& drag) {
            camera.position = cam0.position + (camera.getForward() * drag.y) * sensitivityB;
          }) 
  {}

void ManualCameraController::update(const glm::vec2& cursorPos, const ThreeButtonMouseState& mouseState, float deltaTime) {
  leftDragHelper.checkDragging(mouseState, cursorPos);
  middleDragHelper.checkDragging(mouseState, cursorPos);
  rightDragHelper.checkDragging(mouseState, cursorPos);

  static glm::vec3 pos0{};
  static glm::vec3 tar0{};
  static glm::vec3 deltaPos{};
  const float cameraSpeed = isKeyPressed(GLFW_KEY_LEFT_SHIFT) ? 0.2f : 2.0f;

  bool anyKeyPressed = false;
  auto keys1 = {GLFW_KEY_Q, GLFW_KEY_E, GLFW_KEY_A, GLFW_KEY_D, GLFW_KEY_W, GLFW_KEY_S};
  for (auto key : keys1)
    anyKeyPressed |= isKeyPressed(key);

  if (!anyKeyPressed) {
    deltaPos = {};
    pos0 = camera.position;
    tar0 = camera.target;
  } else {
    
  }

  // up-down in world-space
  if (isKeyPressed(GLFW_KEY_Q))
    deltaPos -= camera.getUp() * cameraSpeed * deltaTime;
  if (isKeyPressed(GLFW_KEY_E))
    deltaPos += camera.getUp() * cameraSpeed * deltaTime;
  // left-right on camera-plane
  if (isKeyPressed(GLFW_KEY_A))
    deltaPos += -camera.getRight() * cameraSpeed * deltaTime;
  if (isKeyPressed(GLFW_KEY_D))
    deltaPos += camera.getRight() * cameraSpeed * deltaTime;
  // forward-backward
  if (isKeyPressed(GLFW_KEY_W))
    deltaPos += camera.getForward() * cameraSpeed * deltaTime;
  if (isKeyPressed(GLFW_KEY_S))
    deltaPos += -camera.getForward() * cameraSpeed * deltaTime;

  camera.position = pos0 + deltaPos;
  camera.target = tar0 + deltaPos;
}
}  // namespace ws