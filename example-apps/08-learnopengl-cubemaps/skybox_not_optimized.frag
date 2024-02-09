#version 460

in vec3 v_TexCoords;

layout(binding = 1) uniform samplerCube skybox;

out vec4 FragColor;

void main() {    
    FragColor = texture(skybox, v_TexCoords);
}