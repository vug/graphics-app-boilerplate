struct VertexData {
  vec3 objectPosition;
  vec3 worldPosition;
  vec3 objectNormal;
  vec3 worldNormal;
  vec2 texCoord;
  vec4 color;
  vec4 custom;
};

VertexData fillVertexData(mat4 worldFromObject, vec3 objectPosition, vec3 objectNormal, vec2 texCoord, vec4 color, vec4 custom) {
  VertexData v;
  v.objectPosition = objectPosition;
  v.worldPosition = vec3(worldFromObject * vec4(objectPosition, 1));
  v.objectNormal = objectNormal;
  v.worldNormal = mat3(transpose(inverse(worldFromObject))) * objectNormal;
  v.texCoord = texCoord;
  v.color = color;
  v.custom = custom;  
  return v;
}