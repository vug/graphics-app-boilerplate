struct PointLight {
  vec3 position;
  float intensity;
  //
  vec3 color;
  float _pad0;
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
  vec3 fragToLightReflected = reflect(fragToLightN, normal);
  vec3 fragToEyeN = normalize(eyePos - position);
  float specular = pow(max(dot(fragToEyeN, fragToLightReflected), 0.0), coeff);
  return light.intensity * light.color * specular;
}

vec3 illuminate(PointLight light, vec3 position, vec3 normal, vec3 eyePos, float specCoeff) {
  vec3 diffuse = illuminateDiffuse(light, position, normal);
  vec3 specular = illuminateSpecular(light, position, normal, eyePos, specCoeff);
  return diffuse + specular;
}