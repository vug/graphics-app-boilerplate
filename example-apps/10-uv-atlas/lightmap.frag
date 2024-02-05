#version 460
#extension GL_ARB_shading_language_include : require

#include "/lib/VertexData.glsl"
#include "/lib/SceneUniforms.glsl"

in VertexData vertexData;

layout(binding = 0) uniform sampler2D diffuseTex;
layout(binding = 1) uniform sampler2D lightmapTex;

layout (location = 0) out vec4 outColor;

void main() {
  vec2 uv2 = vec2(vertexData.texCoord2.x, vertexData.texCoord2.y);
  //vec2 uv2 = vec2(vertexData.texCoord2.x, 1 - vertexData.texCoord2.y);
  vec3 diffuse = texture(diffuseTex, vertexData.texCoord).rgb;
  vec3 light = texture(lightmapTex, uv2).rgb;
  //vec3 light = pow(texture(lightmapTex, uv2).rgb, vec3(4.0));
  outColor = vec4(diffuse * light, 1);
}