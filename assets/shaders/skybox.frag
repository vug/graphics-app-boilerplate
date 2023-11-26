#version 460

in vec3 v_TexCoords;

uniform samplerCube skybox;

out vec4 FragColor;

void main() {    
    FragColor = texture(skybox, v_TexCoords);
    //FragColor = vec4(1, 0, 0, 1);
}