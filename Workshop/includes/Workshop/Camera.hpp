#pragma once

#include <glm/gtx/quaternion.hpp>
#include <glm/mat4x4.hpp>
#include <glm/vec2.hpp>

#include <functional>
#include <numbers>

namespace ws {
class CameraView {
 public:  // TODO: make private again (?)
  glm::vec3 position{};

 public:
  const glm::vec3& getPosition() const;
  virtual glm::vec3 getDirection() const = 0;

  glm::mat4 getViewFromWorld() const;
  glm::vec3 getForward() const;
  glm::vec3 getUp() const;
  glm::vec3 getRight() const;
};

class CameraProjection {
 public:
  float nearClip{0.01f};
  float farClip{100.0f};
  float aspectRatio{1.f};
  virtual glm::mat4 getProjectionFromView() const = 0;
};

class Camera : public CameraView, public CameraProjection {
  glm::mat4 getProjectionFromWorld() const;
};

// 3 position + 3 orientation = 6 DoF camera.
// Orientation can be stated as an unit axis and an angle, or more concise as yaw/pitch/roll
class Camera3DView : public CameraView {
 public:  // TODO: make private again
  float pitch{};
  float yaw{std::numbers::pi_v<float> * 0.5f};
  float roll{};

 public:
  virtual glm::vec3 getDirection() const final;

  friend class Camera3DViewAutoOrbitingController;
};

class PerspectiveCameraProjection : public CameraProjection {
 public:  // TODO: make private again
  float fov{50.f};

 public:
  virtual glm::mat4 getProjectionFromView() const final;
};

class OrthographicCameraProjection : public CameraProjection {
 private:
  float size{5.0f};

 public:
  virtual glm::mat4 getProjectionFromView() const final;
};

class PerspectiveCamera3D : public Camera3DView,
                            public PerspectiveCameraProjection {};

class OrthographicCamera3D : public Camera3DView,
                             public OrthographicCameraProjection {};

class AutoOrbitingCamera3DViewController {
 private:
  Camera3DView& cameraView;
  float phi{};
  float deltaPhi{};

 public:
  float theta{std::numbers::pi_v<float> * 0.2f};
  float phi0{};
  float radius = 3.f;
  float speed = 1.f;

 public:
  AutoOrbitingCamera3DViewController(Camera3DView& cameraView);
  void update(float deltaTime);
};

// A state machine that keeps track of mouse dragging input by the given mouse button.
// calls given callbacks at state changes
class DragHelper {
 private:
  int glfwDragButton{};
  std::function<void()> onEnterDraggingCallback;
  std::function<void(const glm::vec2& drag)> onBeingDraggedCallback;
  // state
  bool isBeingDragged{};
  bool isBeingPressed{};
  glm::vec2 cursor0{};

 public:
  DragHelper(int glfwDragButton, std::function<void()> onEnterDraggingCallback, std::function<void(const glm::vec2& drag)> onBeingDraggedCallback);
  // update function to call every frame
  void checkDragging(int glfwInputButton, const glm::vec2& cursorPos);
};

// First Person Camera mechanism with state machine for dragging
class ManualCamera3DViewController {
 private:
  Camera3DView& cameraView;

  glm::vec3 pos0{};
  float pitch0{};
  float yaw0{};
  DragHelper rightDragHelper;
  DragHelper middleDragHelper;

  const float sensitivity = 0.005f;   // look around sensitivity
  const float sensitivityB = 0.005f;  // pan sensitivity
 public:
  ManualCamera3DViewController(Camera3DView& cameraView);
  void update(const glm::vec2& cursorPos, int glfwInputButton);
};
}  // namespace ws