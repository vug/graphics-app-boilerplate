#version 460

// TODO: get from DefaultVertexAttributes.glsl
layout(location = 0) in vec3 a_Position;
layout(location = 1) in vec2 a_TexCoord;
layout(location = 2) in vec2 a_TexCoord2;
layout(location = 3) in vec3 a_Normal;
layout(location = 4) in vec4 a_Color;
layout(location = 5) in vec4 a_Custom;
// layout(location = 6) in int meshId;

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
  v.objectPosition = a_Position;
  v.worldPosition = vec3(u_WorldFromObject * vec4(a_Position, 1));
  v.objectNormal = a_Normal;
  v.worldNormal = mat3(transpose(inverse(u_WorldFromObject))) * a_Normal;
  v.texCoord = a_TexCoord;
  v.texCoord2 = a_TexCoord2;
  gl_Position = u_ProjectionFromView * u_ViewFromWorld * u_WorldFromObject * vec4(a_Position, 1.0);
}