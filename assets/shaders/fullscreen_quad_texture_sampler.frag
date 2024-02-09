#version 460

in VertexData {
  vec2 uv;
} v;

layout(binding = 0) uniform sampler2D screenTexture;

layout (location = 0) out vec4 outColor;

void main () { 
  //outColor = vec4(v.uv.x, v.uv.y, 0, 1.0); 
  //outColor = texture(screenTexture, v.uv).a > 0.5 ? vec4(1,0,0,1) : vec4(0,0,1,1);
  outColor.rgba = texture(screenTexture, v.uv).rgba;
}