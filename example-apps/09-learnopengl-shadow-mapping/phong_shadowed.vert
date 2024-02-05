#version 460
#extension GL_ARB_shading_language_include : require

#include "/lib/DefaultVertexAttributes.glsl"
#include "/lib/VertexData.glsl"
#include "/lib/SceneUniforms.glsl"

uniform mat4 u_WorldFromObject;
uniform mat4 u_LightSpaceMatrix;

out VertexData vertexData;
out vec4 FragPosLightSpace;

void main() {
  vertexData = fillVertexData(u_WorldFromObject, a_Position, a_Normal, a_TexCoord, a_TexCoord2, a_Color, a_Custom);
  FragPosLightSpace = u_LightSpaceMatrix * vec4(vertexData.worldPosition, 1.0);
  gl_Position = su.u_ProjectionFromView * su.u_ViewFromWorld * vec4(vertexData.worldPosition, 1);
}