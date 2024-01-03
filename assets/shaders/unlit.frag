#version 460

in vec2 v_TexCoords;
in vec2 v_TexCoords2;

uniform sampler2D mainTex;
uniform vec4 u_Color = vec4(1.0, 1.0, 1.0, 1.0);

layout (location = 0) out vec4 outColor;

void main() {
  vec2 uv2Inv = vec2(v_TexCoords2.x, 1 - v_TexCoords2.y);
  //vec3 tex = texture(mainTex, uv2Inv).rgb;
  vec3 tex = texture(mainTex, v_TexCoords).rgb;
  //outColor = vec4(v_TexCoords.x, v_TexCoords.y, 0, 1); // debug uv
  outColor = vec4(tex, 1.0) * u_Color;
}