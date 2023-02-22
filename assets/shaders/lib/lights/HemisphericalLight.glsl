struct HemisphericalLight {
  vec3 northColor;
  vec3 southColor;
  float intensity;
};

uniform HemisphericalLight hemisphericalLight;

vec3 illuminate(HemisphericalLight light, vec3 normal) {
  const vec3 northDir = vec3(0, 1, 0);
  float m = dot(northDir, normal);
  vec3 lightColor = mix(light.southColor, light.northColor, map(m, -1, 1, 0, 1));
  return light.intensity * lightColor;
}