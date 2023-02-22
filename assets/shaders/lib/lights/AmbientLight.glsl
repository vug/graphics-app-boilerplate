struct AmbientLight {
  vec3 color;
};

uniform AmbientLight ambientLight;

vec3 illuminate(AmbientLight light) {
  return light.color;
}