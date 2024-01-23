#version 460

in VertexData {
  vec2 uv;
} v;

uniform sampler2D screenTexture;
uniform bool u_IsHorizontal = true;
uniform float u_Weights[5] = float[] (0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216);

layout (location = 0) out vec4 outColor;

void main () { 
  vec2 tex_offset = 1.0 / textureSize(screenTexture, 0); // gets size of single texel
  vec3 result = texture(screenTexture, v.uv).rgb * u_Weights[0]; // current fragment's contribution
  if (u_IsHorizontal) {
    for(int i = 1; i < 5; ++i) {
      result += texture(screenTexture, v.uv + vec2(tex_offset.x * i, 0.0)).rgb * u_Weights[i];
      result += texture(screenTexture, v.uv - vec2(tex_offset.x * i, 0.0)).rgb * u_Weights[i];
    }
  }
  else {
    for(int i = 1; i < 5; ++i) {
      result += texture(screenTexture, v.uv + vec2(0, tex_offset.y * i)).rgb * u_Weights[i];
      result += texture(screenTexture, v.uv - vec2(0, tex_offset.y * i)).rgb * u_Weights[i];
    }
  }

  outColor = vec4(result, 1);
}