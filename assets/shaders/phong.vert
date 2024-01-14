#version 460

// TODO: get from DefaultVertexAttributes.glsl
layout(location = 0) in vec3 a_Position;
layout(location = 1) in vec2 a_TexCoord;
layout(location = 2) in vec2 a_TexCoord2;
layout(location = 3) in vec3 a_Normal;
layout(location = 4) in vec4 a_Color;
layout(location = 5) in vec4 a_Custom;

// TODO: get from SceneUniforms.glsl
uniform mat4 u_ViewFromWorld = mat4(1);
uniform mat4 u_ProjectionFromView = mat4(1);
//uniform mat4 u_LightSpaceMatrix = mat4(1);
// Object uniforms
uniform mat4 u_WorldFromObject = mat4(1);

// TODO: get from VertexData.glsl
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
out VertexData vertexData;

VertexData fillVertexData(mat4 worldFromObject, vec3 objectPosition, vec3 objectNormal, vec2 texCoord, vec2 texCoord2, vec4 color, vec4 custom) {
  VertexData v;
  v.objectPosition = objectPosition;
  v.worldPosition = vec3(worldFromObject * vec4(objectPosition, 1));
  v.objectNormal = objectNormal;
  v.worldNormal = mat3(transpose(inverse(worldFromObject))) * objectNormal;
  v.texCoord = texCoord;
  v.texCoord2 = texCoord2;
  v.color = color;
  v.custom = custom;  
  return v;
}

void main() {
  vertexData = fillVertexData(u_WorldFromObject, a_Position, a_Normal, a_TexCoord, a_TexCoord2, a_Color, a_Custom);
  gl_Position = u_ProjectionFromView * u_ViewFromWorld * vec4(vertexData.worldPosition, 1);
}