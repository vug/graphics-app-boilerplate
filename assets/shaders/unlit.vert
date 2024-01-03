#version 430

uniform mat4 u_WorldFromObject;
uniform mat4 u_ViewFromWorld;
uniform mat4 u_ProjectionFromView;

layout(location = 0) in vec3 a_Position;
layout(location = 1) in vec2 a_TexCoords;
layout(location = 2) in vec2 a_TexCoords2;

out vec2 v_TexCoords;
out vec2 v_TexCoords2;

void main() {
    gl_Position = u_ProjectionFromView * u_ViewFromWorld * u_WorldFromObject * vec4(a_Position, 1.0);
    v_TexCoords = a_TexCoords;
    v_TexCoords2 = a_TexCoords2;
}