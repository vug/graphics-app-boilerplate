#version 460

layout (location = 0) in vec3 a_Position;

uniform mat4 u_LightSpaceMatrix;
uniform mat4 u_WorldFromObject;

void main() {
    gl_Position = u_LightSpaceMatrix * u_WorldFromObject * vec4(a_Position, 1.0);
}