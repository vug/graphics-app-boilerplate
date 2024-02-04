#version 460
#extension GL_ARB_shading_language_include : require

#include "/lib/DefaultVertexAttributes.glsl"
#include "/lib/SceneUniforms.glsl"

uniform mat4 u_WorldFromObject;

void main() {
  gl_Position = su.u_ProjectionFromView * su.u_ViewFromWorld * u_WorldFromObject * vec4(a_Position, 1.0);
}