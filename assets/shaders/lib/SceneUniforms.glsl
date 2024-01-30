#include "/lib/lights/AmbientLight.glsl"
#include "/lib/lights/PointLights.glsl"
#include "/lib/lights/DirectionalLights.glsl"

#define MAX_POINT_LIGHTS 8
#define MAX_DIRECTIONAL_LIGHTS 4

layout(std140, binding = 1) uniform SceneUniforms {
  mat4 u_ProjectionFromView;
  mat4 u_ViewFromWorld;
  //
  vec3 u_CameraPosition;
  float _pad0;
  //
  AmbientLight ambientLight;
  //
  vec3 _pad1;
  int numPointLights;
  //
  PointLight pointLights[MAX_POINT_LIGHTS];
  //
  vec3 _pad2;
  int numDirectionalLights;
  //
  DirectionalLight directionalLights[MAX_DIRECTIONAL_LIGHTS];
} su;