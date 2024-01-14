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
uniform int u_ShadingModel = 2;
uniform int u_MeshId = -1;

layout(location = 0) out vec4 FragColor;
layout(location = 1) out int MeshId;

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

  MeshId = u_MeshId;

  switch(u_ShadingModel) {
    // Position in Object-Space
    case 0: {
      FragColor = vec4(v.objectPosition, 1);
      return;
    }

    // Position in World-Space
    case 1: {
      FragColor = vec4(v.worldPosition, 1);
      return;
    }

    // UV1
    case 2: {
      FragColor = vec4(v.texCoord.x, v.texCoord.y, 0, 1);
      return;
    }

    // UV2
    case 3: {
      FragColor = vec4(v.texCoord2.x, v.texCoord2.y, 0, 1);
      return;
    }

    // Normal in Object-Space
    case 4: {
      FragColor = vec4(objectNormal * 0.5 + 0.5, 1);
      return;
    }

    // Normal in World-Space
    case 5: {
      FragColor = vec4(worldNormal * 0.5 + 0.5, 1);
      return;
    }

    // Front vs Back faces
    case 6: {
      FragColor = gl_FrontFacing ? vec4(1, 0, 0, 1) : vec4(0, 0, 1, 1);
      return;
    }

    // First texture
    case 7: {
      FragColor = vec4(mainTexColor, 1);
      return;
    }

    // Second texture
    case 8: {
      FragColor = vec4(secondTexColor, 1);
      return;
    }

    // Depth (ortographics projection camera)
    case 9: {
      FragColor = vec4(vec3(gl_FragCoord.z), 1);
      return;
    }

    // Depth (perspective projection camera)
    case 10: {
      float depthViz = LinearizeDepth(gl_FragCoord.z, near, far) / far;
      FragColor = vec4(vec3(depthViz), 1);
      return;
    }

    // MeshIds
    case 11: {
      FragColor = vec4(vec3(u_MeshId / 10.f), 1); // TODO: improve to random color
      return;
    }
  }
}