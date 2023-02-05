#version 300 es
#extension GL_EXT_separate_shader_objects : enable
precision mediump float;

layout (location = 0) in vec3 fragColor;

layout (location = 0) out vec4 outColor;

void main () { 
  outColor = vec4 (fragColor, 1.0); 
}