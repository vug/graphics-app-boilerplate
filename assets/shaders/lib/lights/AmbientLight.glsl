struct AmbientLight {
  vec3 color;
  float _pad0;
};

vec3 illuminate(AmbientLight light) {
  return light.color;
}