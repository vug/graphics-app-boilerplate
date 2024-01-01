#version 460

struct VertexData {
  vec3 objectPosition;
  vec3 worldPosition;
  vec3 objectNormal;
  vec3 worldNormal;
  vec2 texCoord;
  vec2 texCoord2;
};

in VertexData v;

out vec4 FragColor;

void main() {
  const vec3 objectNormal = normalize(v.objectNormal);
  const vec3 worldNormal = normalize(v.worldNormal);

  // objectPos
  //FragColor = vec4(v.objectPosition, 1);
  // worldPos
  //FragColor = vec4(v.worldPosition, 1);
  // uv1
  //FragColor = vec4(v.texCoord.x, v.texCoord.y, 0, 1);
  // uv2
  //FragColor = vec4(v.texCoord2.x, v.texCoord2.y, 0, 1);
  // objectNorm
  //FragColor = vec4(objectNormal * 0.5 + 0.5, 1);
  // worldNorm
  //FragColor = vec4(worldNormal * 0.5 + 0.5, 1);
  // front-back faces
  FragColor = gl_FrontFacing ? vec4(1, 0, 0, 1) : vec4(0, 0, 1, 1);
}