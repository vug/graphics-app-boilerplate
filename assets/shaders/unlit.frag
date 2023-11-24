#version 460

in vec2 v_TexCoords;

uniform sampler2D texture1;

layout (location = 0) out vec4 outColor;

void main() {
  vec3 tex = texture(texture1, v_TexCoords).rgb;
  //outColor = vec4(v_TexCoords.x, v_TexCoords.y, 0, 1); // debug uv
  outColor = vec4(tex, 1.0);
}