#version 460

#extension GL_ARB_shading_language_include : require
#include "/lib/VertexData.glsl"
#include "/lib/SceneUniforms.glsl"

in VertexData vertexData;

// Material uniforms
layout(binding = 0) uniform sampler2D diffuseTexture;
layout(binding = 1) uniform sampler2D specularTexture;

out vec4 FragColor;

void main() {
  const float specCoeff = 32.f;

  const vec3 surfPos = vertexData.worldPosition;
  const vec3 normal = normalize(vertexData.worldNormal);
  vec3 diffuseColor = texture(diffuseTexture, vertexData.texCoord).rgb;
  vec3 specularColor = texture(specularTexture, vertexData.texCoord).rgb;

  vec3 ambient = illuminate(su.ambientLight);

  vec3 directionalDiffuse = vec3(0);
  vec3 directionalSpecular = vec3(0);
  for (int i = 0; i < su.numDirectionalLights; i++) {
    directionalDiffuse += illuminateDiffuse(su.directionalLights[i], normal);
    directionalSpecular += illuminateSpecular(su.directionalLights[i], surfPos, normal, su.u_CameraPosition, specCoeff);
  }

  vec3 pointDiffuse = vec3(0);
  vec3 pointSpecular = vec3(0);
  for (int i = 0; i < su.numPointLights; i++) {
    pointDiffuse += illuminateDiffuse(su.pointLights[i], surfPos, normal);
    pointSpecular += illuminateSpecular(su.pointLights[i], surfPos, normal, su.u_CameraPosition, specCoeff);
  }

  vec3 color = ambient + diffuseColor * (directionalDiffuse + pointDiffuse) + specularColor * (directionalSpecular + pointSpecular);
  FragColor = vec4(color, 1);
}