#version 460
layout(location = 0) in vec3 a_Position;

uniform mat4 u_ViewFromWorld;
uniform mat4 u_ProjectionFromView;

out vec3 v_TexCoords;

void main() {
    v_TexCoords = a_Position;
    gl_Position = u_ProjectionFromView * u_ViewFromWorld * vec4(a_Position, 1.0);
}