#version 460

in vec2 v_TexCoords;

uniform sampler2D mainTex;
uniform vec4 u_Color = vec4(1.0, 1.0, 1.0, 1.0);

layout (location = 0) out vec4 outColor;

void main() {
  vec3 tex = texture(mainTex, v_TexCoords).rgb;
  outColor = vec4(tex, 1.0) * u_Color;
}