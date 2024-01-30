struct DirectionalLight {
  vec3 position;
  float intensity;
  //
  vec3 direction;
  float _pad0;
  //
  vec3 color;
  float _pad1;
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