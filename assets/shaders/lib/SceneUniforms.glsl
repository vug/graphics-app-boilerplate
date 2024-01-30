#include "/lib/lights/PointLights.glsl"

#define MAX_POINT_LIGHTS 8

layout(std140, binding = 1) uniform SceneUniforms {
  mat4 u_ProjectionFromView;
  mat4 u_ViewFromWorld;
  //
  vec3 u_CameraPosition;
  float _pad0;
  //
  vec3 _pad1;
  int numPointLights;
  //
  PointLight pointLights[MAX_POINT_LIGHTS];
} su;