#version 460

#extension GL_ARB_shading_language_include : require
#include "/lib/DefaultVertexAttributes.glsl"
#include "/lib/VertexData.glsl"

layout(std140, binding = 1) uniform SceneUniforms {
  mat4 u_ProjectionFromView;
  mat4 u_ViewFromWorld;
  vec3 u_CameraPosition;
} su;


uniform mat4 u_WorldFromObject = mat4(1);

out VertexData vertexData;

void main() {
  vertexData = fillVertexData(u_WorldFromObject, a_Position, a_Normal, a_TexCoord, a_TexCoord2, a_Color, a_Custom);
  gl_Position = su.u_ProjectionFromView * su.u_ViewFromWorld * vec4(vertexData.worldPosition, 1);
}