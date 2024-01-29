layout(std140, binding = 1) uniform SceneUniforms {
  mat4 u_ProjectionFromView;
  mat4 u_ViewFromWorld;
  vec3 u_CameraPosition;
} su;