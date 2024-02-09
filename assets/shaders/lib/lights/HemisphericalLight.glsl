struct HemisphericalLight {
  vec3 northColor;
  float intensity;
  vec3 southColor;
  float _pad0;
};

float map(float value, float min1, float max1, float min2, float max2) {
  return min2 + (value - min1) * (max2 - min2) / (max1 - min1);
}

vec3 illuminateDiffuse(HemisphericalLight light, vec3 normal) {
  const vec3 northDir = vec3(0, 1, 0);
  float m = dot(northDir, normal);
  vec3 lightColor = mix(light.southColor, light.northColor, map(m, -1, 1, 0, 1));
  return light.intensity * lightColor;
}

vec3 illuminateSpecular(HemisphericalLight light, vec3 position, vec3 normal, vec3 eyePos, float coeff) {
  const vec3 northDir = vec3(0, 1, 0);
  vec3 fragToLightN = normalize(northDir);
  vec3 fragToLightReflected = reflect(-fragToLightN, normal);
  vec3 fragToEyeN = normalize(eyePos - position);
  float specular = pow(max(dot(fragToEyeN, fragToLightReflected), 0.0), coeff);
  return light.intensity * light.northColor * specular;
}

vec3 illuminate(HemisphericalLight light, vec3 position, vec3 normal, vec3 eyePos, float specCoeff) {
  vec3 diffuse = illuminateDiffuse(light, normal);
  vec3 specular = illuminateSpecular(light, position, normal, eyePos, specCoeff);
  return diffuse + specular;
}