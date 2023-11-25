#version 460

in VertexData {
  vec2 uv;
} v;

uniform sampler2D screenTexture;

layout (location = 0) out vec4 outColor;

ivec2[8] neighbors = {
 ivec2(-1,  1), ivec2(0,  1), ivec2(1,  1),
 ivec2(-1,  0),               ivec2(1,  0),
 ivec2(-1, -1), ivec2(0, -1), ivec2(1, -1)
};

void main () {
  int cnt = 0;
  for (int i = 0; i < 8; i ++) {
    vec4 texelValue = texelFetch(screenTexture, ivec2(gl_FragCoord.xy) + neighbors[i], 0);
    if (texelValue.r > 0)
      cnt++;
  }
  // if (cnt > 0) // just grow
  if (cnt > 0 && cnt < 8) // outline
    outColor = vec4(1, 1, 0, 1);
  else
    outColor = vec4(1, 1, 1, 0);
}