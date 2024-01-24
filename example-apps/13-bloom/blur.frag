#version 460

in VertexData {
  vec2 uv;
} v;

uniform sampler2D screenTexture;
uniform bool u_IsHorizontal = true;
#define SAMPLE_COUNT 5
uniform float u_Weights[SAMPLE_COUNT] = float[] (0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216);
//uniform float u_Weights[SAMPLE_COUNT] = float[] (0.23402612, 0.19756781, 0.11886985, 0.05097192, 0.01557736);
//#define SAMPLE_COUNT 11
//uniform float u_Weights[SAMPLE_COUNT] = float[] (0.07896718, 0.07757572, 0.07354675, 0.06729139, 0.05941745, 0.05063221, 0.0416388 , 0.03304668, 0.02531139, 0.01870952, 0.0133465);

layout (location = 0) out vec4 outColor;

void main () { 
  vec2 tex_offset = 1.0 / textureSize(screenTexture, 0); // gets size of single texel
  vec3 result = texture(screenTexture, v.uv).rgb * u_Weights[0]; // current fragment's contribution
  if (u_IsHorizontal) {
    for(int i = 1; i < SAMPLE_COUNT; ++i) {
      result += texture(screenTexture, v.uv + vec2(tex_offset.x * i, 0.0)).rgb * u_Weights[i];
      result += texture(screenTexture, v.uv - vec2(tex_offset.x * i, 0.0)).rgb * u_Weights[i];
    }
  }
  else {
    for(int i = 1; i < SAMPLE_COUNT; ++i) {
      result += texture(screenTexture, v.uv + vec2(0, tex_offset.y * i)).rgb * u_Weights[i];
      result += texture(screenTexture, v.uv - vec2(0, tex_offset.y * i)).rgb * u_Weights[i];
    }
  }

  outColor = vec4(result, 1);
}