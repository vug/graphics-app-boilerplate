#version 460

layout (location = 0) in vec3 a_Position;
layout (location = 1) in vec2 a_TexCoords;
layout (location = 2) in vec2 a_TexCoords2;
layout (location = 3) in vec3 a_Normal;

out VS_OUT {
    vec3 FragPos;
    vec3 Normal;
    vec2 TexCoords;
    vec4 FragPosLightSpace;
} vs_out;

uniform mat4 u_WorldFromObject;
uniform mat4 u_ViewFromWorld;
uniform mat4 u_ProjectionFromView;
uniform mat4 u_LightSpaceMatrix;

void main()
{
    vs_out.FragPos = vec3(u_WorldFromObject * vec4(a_Position, 1.0));
    vs_out.Normal = transpose(inverse(mat3(u_WorldFromObject))) * a_Normal;
    vs_out.TexCoords = a_TexCoords;
    vs_out.FragPosLightSpace = u_LightSpaceMatrix * vec4(vs_out.FragPos, 1.0);
    gl_Position = u_ProjectionFromView * u_ViewFromWorld * u_WorldFromObject * vec4(a_Position, 1.0);
}