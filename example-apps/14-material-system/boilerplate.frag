#version 460

#extension GL_ARB_shading_language_include : require
#include "/lib/VertexData.glsl"
#include "/lib/SceneUniforms.glsl"

in VertexData vertexData;

uniform vec3 color1 = vec3(0.8, 0.8, 0.8);
uniform vec3 color2 = vec3(0.2, 0.2, 0.2);
uniform int numCells = 4;
layout(binding = 3) uniform sampler2D mainTex;
layout(binding = 7) uniform sampler2D secondTex;

out vec4 FragColor;

void main() {
  const vec3 objectNormal = normalize(vertexData.objectNormal);
  const vec3 worldNormal = normalize(vertexData.worldNormal);
  const vec3 mainTexColor = texture(mainTex, vertexData.texCoord).rgb;
  const vec3 secondTexColor = texture(secondTex, vertexData.texCoord).rgb;

  // objectPos
  //FragColor = vec4(vertexData.objectPosition, 1);
  // worldPos
  //FragColor = vec4(vertexData.worldPosition, 1);
  // uv1
  //FragColor = vec4(vertexData.texCoord.x, vertexData.texCoord.y, 0, 1);
  // uv2
  //FragColor = vec4(vertexData.texCoord2.x, vertexData.texCoord2.y, 0, 1);
  // objectNorm
  //FragColor = vec4(objectNormal * 0.5 + 0.5, 1);
  // worldNorm
  FragColor = vec4(worldNormal * 0.5 + 0.5, 1);
  // front-back faces
  //FragColor = gl_FrontFacing ? vec4(1, 0, 0, 1) : vec4(0, 0, 1, 1);
  // first texture
  //FragColor = vec4(mainTexColor, 1);
  // second texture
  //FragColor = vec4(secondTexColor, 1);

  vec2 ij = floor(vertexData.texCoord * numCells);
  vec2 uv = fract(vertexData.texCoord * numCells);
  vec3 color = mod(ij.x + ij.y, 2) == 0 ? color1 : color2;
  color *= mainTexColor;
  FragColor = vec4(color, 1);
}