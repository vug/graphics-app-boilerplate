#version 460

vec2 positions[4] = vec2[](vec2 (-.5, -.5), vec2 (.5, -.5), vec2 (.5, .5), vec2(-.5, .5));
int indices[6] = {0, 1, 2, 0, 2, 3};

uniform mat4 u_ViewFromWorld = mat4(1);
uniform mat4 u_ProjectionFromView = mat4(1);
uniform vec3 u_CameraPosition = vec3(0, 0, -5);

layout(location = 0) out vec2 uv;
layout(location = 1) out vec2 p2D;

void main () {
  int ix = indices[gl_VertexID];
  uv = positions[ix] + 0.5;
  p2D = 1000 * positions[ix] + u_CameraPosition.xz;

	// xz-plane
	gl_Position = u_ProjectionFromView * u_ViewFromWorld * vec4(p2D.x, 0.0, p2D.y, 1.0);
}