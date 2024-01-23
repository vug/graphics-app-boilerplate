#version 460

in VertexData {
  vec2 uv;
} v;

uniform sampler2D screenTexture;
uniform float u_LuminanceThreshold = 0.75;

layout (location = 0) out vec4 outColor;

void main () { 
  vec3 col = texture(screenTexture, v.uv).rgb;
  float luminance = 0.2126 * col.r + 0.7152 * col.g + 0.0722 * col.b;
  outColor = (luminance > u_LuminanceThreshold) ? vec4(col, 1) : vec4(vec3(0), 1);
}