#version 460

layout(location = 0) in vec2 uv;
layout(location = 1) in vec2 p2D;

uniform vec3 u_CameraPosition = vec3(0, 0, -5);

out vec4 FragColor;

void main() {
  //FragColor = vec4(1, 0, 0, 1);

  //FragColor = vec4(uv.x, uv.y, 0, 1);

  const vec2 cellPos = fract(p2D);
  FragColor = vec4(cellPos, 0, 1);

  if (cellPos.x < 0.05 || cellPos.y < 0.05) {
    FragColor = vec4(1, 1, 1, 1);
  } else {
    discard;
  }
}