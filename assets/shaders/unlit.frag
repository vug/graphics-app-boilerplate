#version 460
#extension GL_ARB_shading_language_include : require

#include "/lib/VertexData.glsl"
#include "/lib/SceneUniforms.glsl"

in VertexData vertexData;

uniform vec4 u_Color = vec4(1.0, 1.0, 1.0, 1.0);
layout (binding = 0) uniform sampler2D mainTex;

layout (location = 0) out vec4 outColor;

void main() {
  const vec3 tex = texture(mainTex, vertexData.texCoord).rgb;
  outColor = vec4(tex, 1.0) * u_Color;
}