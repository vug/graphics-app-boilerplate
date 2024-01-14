#version 460

struct VertexData {
  vec3 objectPosition;
  vec3 worldPosition;
  vec3 objectNormal;
  vec3 worldNormal;
  vec2 texCoord;
  vec2 texCoord2;
};
in VertexData v;

layout(binding = 0) uniform sampler2D mainTex;
layout(binding = 1) uniform sampler2D secondTex;
uniform vec2 u_CameraNearFar; // .x: near, .y: far

out vec4 FragColor;

float LinearizeDepth(float depth, float near, float far) {
    float z = depth * 2.0 - 1.0; // back to NDC 
    return (2.0 * near * far) / (far + near - z * (far - near));	
}

void main() {
  const vec3 objectNormal = normalize(v.objectNormal);
  const vec3 worldNormal = normalize(v.worldNormal);
  const vec3 mainTexColor = texture(mainTex, v.texCoord).rgb;
  const vec3 secondTexColor = texture(secondTex, v.texCoord).rgb;
  const float near = u_CameraNearFar.x;
  const float far = u_CameraNearFar.y;

  // Position in Object-Space
  //FragColor = vec4(v.objectPosition, 1);
  
  // Position in World-Space
  //FragColor = vec4(v.worldPosition, 1);
  
  // UV1
  FragColor = vec4(v.texCoord.x, v.texCoord.y, 0, 1);
  
  // UV2
  //FragColor = vec4(v.texCoord2.x, v.texCoord2.y, 0, 1);
  
  // Normal in Object-Space
  //FragColor = vec4(objectNormal * 0.5 + 0.5, 1);

  // Normal in World-Space
  //FragColor = vec4(worldNormal * 0.5 + 0.5, 1);

  // Front vs Back faces
  //FragColor = gl_FrontFacing ? vec4(1, 0, 0, 1) : vec4(0, 0, 1, 1);

  // First texture
  //FragColor = vec4(mainTexColor, 1);

  // Second texture
  //FragColor = vec4(secondTexColor, 1);
  
  // Depth (ortographics projection camera)
  //FragColor = vec4(vec3(gl_FragCoord.z), 1);
  
  // Depth (perspective projection camera)
  //float depthViz = LinearizeDepth(gl_FragCoord.z, near, far) / far;
  //FragColor = vec4(vec3(depthViz), 1);
}