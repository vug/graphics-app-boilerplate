#version 460

layout(location = 0) in vec3 a_ObjectPosition;
layout(location = 1) in vec2 a_TexCoord;
layout(location = 2) in vec2 a_TexCoord2;
layout(location = 3) in vec3 a_ObjectNormal;

struct VertexData {
  vec3 objectPosition;
  vec3 worldPosition;
  vec3 objectNormal;
  vec3 worldNormal;
  vec2 texCoord;
  vec2 texCoord2;
};

uniform mat4 u_WorldFromObject;
uniform mat4 u_ViewFromWorld;
uniform mat4 u_ProjectionFromView;

out VertexData v;

void main() {
  v.objectPosition = a_ObjectPosition;
  v.worldPosition = vec3(u_WorldFromObject * vec4(a_ObjectPosition, 1));
  v.objectNormal = a_ObjectNormal;
  v.worldNormal = mat3(transpose(inverse(u_WorldFromObject))) * a_ObjectNormal;
  v.texCoord = a_TexCoord;
  v.texCoord2 = a_TexCoord2;
  //gl_Position = u_ProjectionFromView * u_ViewFromWorld * u_WorldFromObject * vec4(a_ObjectPosition, 1.0);
  gl_Position = vec4(a_TexCoord2 * 2 - 1, 0, 1);
}