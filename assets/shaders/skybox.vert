#version 460

layout(location = 0) in vec3 a_Position;

uniform mat4 u_ViewFromWorld;
uniform mat4 u_ProjectionFromView;

out vec3 v_TexCoords;

void main() {
    v_TexCoords = a_Position;
    vec4 pos = u_ProjectionFromView * u_ViewFromWorld * vec4(a_Position, 1.0);
    // trick to make all depth values from skybox (a cube of size 1) equal to be 1
    // pos.xyz will be divided by w and z will be depth
    gl_Position = pos.xyww;
}  