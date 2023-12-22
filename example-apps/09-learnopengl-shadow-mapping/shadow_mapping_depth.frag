#version 460

layout (location = 0) out vec4 outColor; // remove

void main() {             
    // gl_FragDepth = gl_FragCoord.z;
    outColor = vec4(gl_FragCoord.z / gl_FragCoord.w, 0, 0, 1);
}