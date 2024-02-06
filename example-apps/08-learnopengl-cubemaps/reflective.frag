#version 460

in vec3 v_WorldPosition;
in vec2 v_TexCoord;
in vec3 v_Normal;

//layout(binding = 0) uniform sampler2D mainTex;
layout(binding = 1) uniform samplerCube skybox;
uniform vec3 u_CameraPos;
uniform float u_RefRefMix = 0.25;
uniform vec4 u_Color = vec4(1, 1, 1, 1);

layout (location = 0) out vec4 outColor;

void main() {
  const vec3 incident = v_WorldPosition - u_CameraPos;
  const vec3 normal = normalize(v_Normal);
  const vec3 reflectedRay = reflect(incident, normal);
  float ratio = 1.00 / 1.52; // refractive index of air is 1. Glass is 1.52
  const vec3 refractedRay = refract(incident, normal, ratio);

  //outColor = vec4(v_Normal, 1);
  //outColor = vec4(normal*0.5 + 0.5, 1);
  //outColor = texture(skybox, normal);
  //outColor = vec4(normalize(u_CameraPos), 1);

  //outColor = texture(skybox, reflectedRay) * texture(mainTex, v_TexCoord);
  outColor = mix(texture(skybox, reflectedRay), texture(skybox, refractedRay), u_RefRefMix) * u_Color;
}