#version 460

layout(location = 0) in vec3 a_Position;
layout(location = 3) in vec3 a_Normal;

struct VertexData {
  vec3 worldPosition;
  vec3 worldNormal;
};

uniform mat4 u_WorldFromObject;
uniform mat4 u_ViewFromWorld;
uniform mat4 u_ProjectionFromView;

out VertexData v;

void main() {
  v.worldPosition = vec3(u_WorldFromObject * vec4(a_Position, 1));
  v.worldNormal = mat3(transpose(inverse(u_WorldFromObject))) * a_Normal;
  //gl_Position = u_ProjectionFromView * u_ViewFromWorld * u_WorldFromObject * vec4(a_Position, 1.0);
}