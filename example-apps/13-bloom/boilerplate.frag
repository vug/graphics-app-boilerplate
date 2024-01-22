#version 460

struct VertexData {
  vec3 objectPosition;
  vec3 worldPosition;
  vec3 objectNormal;
  vec3 worldNormal;
  vec2 texCoord;
  vec2 texCoord2;
  vec4 color;
  vec4 custom;
};
in VertexData vertexData;

layout(binding = 0) uniform sampler2D mainTex;
layout(binding = 1) uniform sampler2D secondTex;
uniform vec3 u_CameraPosition = vec3(0, 0, -5);

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
  //FragColor = vec4(worldNormal * 0.5 + 0.5, 1);
  // front-back faces
  //FragColor = gl_FrontFacing ? vec4(1, 0, 0, 1) : vec4(0, 0, 1, 1);
  // first texture
  //FragColor = vec4(mainTexColor, 1);
  // second texture
  //FragColor = vec4(secondTexColor, 1);

  FragColor = vec4((mainTexColor + secondTexColor + vec3(vertexData.texCoord, 0))/3, 1);
}