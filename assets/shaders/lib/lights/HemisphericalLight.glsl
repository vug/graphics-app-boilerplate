struct HemisphericalLight {
  vec3 northColor;
  float intensity;
  vec3 southColor;
  float _pad0;
};

float map(float value, float min1, float max1, float min2, float max2) {
  return min2 + (value - min1) * (max2 - min2) / (max1 - min1);
}

vec3 illuminate(HemisphericalLight light, vec3 normal) {
  const vec3 northDir = vec3(0, 1, 0);
  float m = dot(northDir, normal);
  vec3 lightColor = mix(light.southColor, light.northColor, map(m, -1, 1, 0, 1));
  return light.intensity * lightColor;
}