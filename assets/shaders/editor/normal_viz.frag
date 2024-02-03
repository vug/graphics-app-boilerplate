#version 460 core

struct GeomData {
  vec4 color;
};
in GeomData g;

out vec4 FragColor;

void main() {
    FragColor = g.color;
} 