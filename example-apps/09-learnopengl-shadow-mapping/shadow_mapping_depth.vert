#version 460

layout (location = 0) in vec3 a_Position;

uniform mat4 lightSpaceMatrix;
uniform mat4 model;

void main() {
    gl_Position = lightSpaceMatrix * model * vec4(a_Position, 1.0);
}