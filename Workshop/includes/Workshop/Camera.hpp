#pragma once

#include "Input.hpp"

#include <glm/gtx/quaternion.hpp>
#include <glm/mat4x4.hpp>
#include <glm/vec2.hpp>

#include <numbers>

#pragma warning(disable : 4250)
namespace ws {
class Camera {
public:
	glm::vec3 position{0.f, 3.f, -5.f};
	glm::vec3 target{0.f, 0.f, 0.f};
	float nearClip{0.01f};
	float farClip{100.0f};
	float aspectRatio{1.f};
	// Perspective Camera
	float fov{50.f};
	// Orthographic Camera
	float orthoSize{5.0f};

	glm::vec3 getDirection() const;
	glm::mat4 getViewFromWorld() const;
	glm::vec3 getForward() const;
	glm::vec3 getUp() const;
	glm::vec3 getRight() const;

	glm::mat4 getProjectionFromView() const;
	glm::mat4 getProjectionFromWorld() const;

	float getPitch() const;
	float getYaw() const;
	float getRoll() const;	
};

class ICamera {
 public:
  virtual const glm::vec3& getPosition() const = 0;
  virtual glm::vec3 getDirection() const = 0;
  virtual glm::mat4 getViewFromWorld() const = 0;
  virtual glm::vec3 getForward() const = 0;
  virtual glm::vec3 getUp() const = 0;
  virtual glm::vec3 getRight() const = 0;

  virtual glm::mat4 getProjectionFromView() const = 0;

  glm::mat4 getProjectionFromWorld() const;
};

// Splitting ICamera into two parts: The View and the Projection
// View is about ViewFromWorld matrix. Near, far, FOV etc. are irrelevant.
class ICameraView : public virtual ICamera {
 public:  // TODO: make private again (?)
  glm::vec3 position{};

 public:
  virtual const glm::vec3& getPosition() const final;
  virtual glm::vec3 getDirection() const = 0;

  virtual glm::mat4 getViewFromWorld() const final;
  virtual glm::vec3 getForward() const final;
  virtual glm::vec3 getUp() const final;
  virtual glm::vec3 getRight() const final;
};

// Splitting ICamera into two parts. The View and Projection
// Projection is about ProjectionFromView matrix. Cam position, direction etc. are irrelevant
class ICameraProjection : public virtual ICamera {
 public:
  float nearClip{0.01f};
  float farClip{100.0f};
  float aspectRatio{1.f};
  virtual glm::mat4 getProjectionFromView() const = 0;
};

// 3 position + 3 orientation = 6 DoF camera view.
// Orientation can be stated as an unit axis and an angle, or more concise as yaw/pitch/roll
class Camera3DView : public ICameraView {
 public:  // TODO: make private again
  float pitch{};
  float yaw{std::numbers::pi_v<float> * 0.5f};
  float roll{};

 public:
  virtual glm::vec3 getDirection() const final;

  friend class Camera3DViewAutoOrbitingController;
};

class PerspectiveCameraProjection : public ICameraProjection {
 public:  // TODO: make private again
  float fov{50.f};

 public:
  virtual glm::mat4 getProjectionFromView() const final;
};

class OrthographicCameraProjection : public ICameraProjection {
 private:
  float size{5.0f};

 public:
  virtual glm::mat4 getProjectionFromView() const final;
};

class PerspectiveCamera3D : public virtual ICamera,
                            public Camera3DView,
                            public PerspectiveCameraProjection {};

class OrthographicCamera3D : public virtual ICamera,
                             public Camera3DView,
                             public OrthographicCameraProjection {};

class AutoOrbitingCameraController {
 private:
  Camera& camera;
  float phi{};
  float deltaPhi{};

 public:
  float theta{std::numbers::pi_v<float> * 0.2f};
  float phi0{};
  float radius = 3.f;
  float speed = 1.f;

 public:
  AutoOrbitingCameraController(Camera& camera);
  void update(float deltaTime);
};

// First Person Camera mechanism with state machine for dragging
class ManualCameraController {
 private:
  Camera& camera;

  Camera cam0;
  DragHelper leftDragHelper;
  DragHelper rightDragHelper;
  DragHelper middleDragHelper;

  const float sensitivity = 0.005f;   // look around sensitivity
  const float sensitivityB = 0.005f;  // pan sensitivity
 public:
  ManualCameraController(Camera& camera);
  void update(const glm::vec2& cursorPos, const ThreeButtonMouseState& mouseState, float deltaTime);
};
}  // namespace ws
#pragma warning(default : 4250)