#version 460

in vec2 v_TexCoords;
in vec2 v_TexCoords2;

layout(binding = 0) uniform sampler2D diffuseTex;
layout(binding = 1) uniform sampler2D lightmapTex;

layout (location = 0) out vec4 outColor;

void main() {
  vec2 uv2 = vec2(v_TexCoords2.x, v_TexCoords2.y);
  //vec2 uv2 = vec2(v_TexCoords2.x, 1 - v_TexCoords2.y);
  vec3 diffuse = texture(diffuseTex, v_TexCoords).rgb;
  vec3 light = texture(lightmapTex, uv2).rgb;
  //vec3 light = pow(texture(lightmapTex, uv2).rgb, vec3(4.0));
  outColor = vec4(diffuse * light, 1);
}