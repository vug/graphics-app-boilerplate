#version 460

layout(location = 0) in vec3 a_Position;
layout(location = 1) in vec2 a_TexCoord;
layout(location = 2) in vec2 a_TexCoord2;
layout(location = 3) in vec3 a_Normal;

uniform mat4 u_WorldFromObject;
uniform mat4 u_ViewFromWorld;
uniform mat4 u_ProjectionFromView;

out vec3 v_WorldPosition;
out vec2 v_TexCoord;
out vec3 v_Normal;

void main() {
    v_WorldPosition = vec3(u_WorldFromObject * vec4(a_Position, 1.0));
    gl_Position = u_ProjectionFromView * u_ViewFromWorld * vec4(v_WorldPosition, 1);

    v_TexCoord = a_TexCoord;

    v_Normal = mat3(transpose(inverse(u_WorldFromObject))) * a_Normal;
}