#version 460

#extension GL_ARB_shading_language_include : require
#include "/lib/VertexData.glsl"
#include "/lib/SceneUniforms.glsl"

in VertexData vertexData;


// get from PointLight.glsl
struct PointLight {
  vec3 position;
  vec3 color;
  float intensity;
};
vec3 illuminateDiffuse(PointLight light, vec3 position, vec3 normal) {
  vec3 fragToLight = light.position - position;
  vec3 fragToLightN = normalize(fragToLight);
  const float fragToLightSquare = dot(fragToLight, fragToLight);
  float diffuse = max(dot(fragToLightN, normal), 0) / fragToLightSquare;
  return light.intensity * light.color * diffuse;
}
vec3 illuminateSpecular(PointLight light, vec3 position, vec3 normal, vec3 eyePos, float coeff) {
  vec3 fragToLightN = normalize(light.position - position);
  vec3 fragToLightReflected = reflect(-fragToLightN, normal);
  vec3 fragToEyeN = normalize(eyePos - position);
  float specular = pow(max(dot(fragToEyeN, fragToLightReflected), 0.0), coeff);
  return light.intensity * light.color * specular;
}
vec3 illuminate(PointLight light, vec3 position, vec3 normal, vec3 eyePos, float specCoeff) {
  vec3 diffuse = illuminateDiffuse(light, position, normal);
  vec3 specular = illuminateSpecular(light, position, normal, eyePos, specCoeff);
  return diffuse + specular;
}

// get from DirectionalLight.glsl
struct DirectionalLight {
  vec3 position;
  vec3 direction;
  vec3 color;
  float intensity;
};
vec3 illuminateDiffuse(DirectionalLight light, vec3 normal) {
  vec3 fragToLightN = normalize(-light.direction);
  return light.intensity * light.color * max(dot(fragToLightN, normal), 0);
}
vec3 illuminateSpecular(DirectionalLight light, vec3 position, vec3 normal, vec3 eyePos, float coeff) {
  vec3 fragToLightN = normalize(-light.direction);
  vec3 fragToLightReflected = reflect(-fragToLightN, normal);
  vec3 fragToEyeN = normalize(eyePos - position);
  float specular = pow(max(dot(fragToEyeN, fragToLightReflected), 0.0), coeff);
  return light.intensity * light.color * specular;
}
vec3 illuminate(DirectionalLight light, vec3 position, vec3 normal, vec3 eyePos, float specCoeff) {
  vec3 diffuse = illuminateDiffuse(light, normal);
  vec3 specular = illuminateSpecular(light, position, normal, eyePos, specCoeff);
  return diffuse + specular;
}

// TODO: get from SceneUniforms.glsl
uniform PointLight pointLight = PointLight(vec3(0, 0, 3), vec3(1, 1, 1), 1.0f);
uniform DirectionalLight directionalLight = DirectionalLight(vec3(1, 1, 1), vec3(-1, -1, -1), vec3(1, 1, 1), 0.5f);
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

  vec3 directionalDiffuse = illuminateDiffuse(directionalLight, normal);
  vec3 directionalSpecular = illuminateSpecular(directionalLight, surfPos, normal, su.u_CameraPosition, specCoeff);

  vec3 pointDiffuse = illuminateDiffuse(pointLight, surfPos, normal);
  vec3 pointSpecular = illuminateSpecular(pointLight, surfPos, normal, su.u_CameraPosition, specCoeff);
  vec3 color = diffuseColor * (directionalDiffuse + pointDiffuse) + specularColor * (directionalSpecular + pointSpecular);
  FragColor = vec4(color, 1);
}