#version 460

layout(location = 0) in vec2 uv;
layout(location = 1) in vec2 p2D;

uniform vec3 u_CameraPosition = vec3(0, 0, -5);

out vec4 FragColor;

void main() {
  //FragColor = vec4(1, 0, 0, 1);
  //FragColor = vec4(uv.x, uv.y, 0, 1);
  FragColor = vec4(fract(p2D), 0, 1);
}