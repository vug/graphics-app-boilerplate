#version 460

struct VertexData {
  vec3 objectPosition;
  vec3 worldPosition;
  vec3 objectNormal;
  vec3 worldNormal;
  vec2 texCoord;
  vec2 texCoord2;
  vec4 color;
  vec4 custom;
};
in VertexData vertexData;

layout(binding = 0) uniform sampler2D mainTex;
layout(binding = 1) uniform sampler2D secondTex;
uniform vec3 u_CameraPosition = vec3(0, 0, -5);

// TODO: get from PointLight.glsl
struct PointLight {
  vec3 position;
  vec3 color;
  float intensity;
};
vec3 illuminateDiffuse(PointLight light, vec3 position, vec3 normal) {
  vec3 fragToLight = light.position - position;
  vec3 fragToLightN = normalize(fragToLight);
  const float fragToLightSquare = dot(fragToLight, fragToLight);
  float diffuse = max(dot(fragToLightN, normal), 0) / fragToLightSquare;
  return light.intensity * light.color * diffuse;
}
vec3 illuminateSpecular(PointLight light, vec3 position, vec3 normal, vec3 eyePos, float coeff) {
  vec3 fragToLightN = normalize(light.position - position);
  vec3 fragToLightReflected = reflect(-fragToLightN, normal);
  vec3 fragToEyeN = normalize(eyePos - position);
  float specular = pow(max(dot(fragToEyeN, fragToLightReflected), 0.0), coeff);
  return light.intensity * light.color * specular;
}
vec3 illuminate(PointLight light, vec3 position, vec3 normal, vec3 eyePos, float specCoeff) {
  vec3 diffuse = illuminateDiffuse(light, position, normal);
  vec3 specular = illuminateSpecular(light, position, normal, eyePos, specCoeff);
  return diffuse + specular;
}

// TODO: get from DirectionalLight.glsl
struct DirectionalLight {
  vec3 position;
  vec3 direction;
  vec3 color;
  float intensity;
};
vec3 illuminateDiffuse(DirectionalLight light, vec3 normal) {
  vec3 fragToLightN = normalize(-light.direction);
  return light.intensity * light.color * max(dot(fragToLightN, normal), 0);
}
vec3 illuminateSpecular(DirectionalLight light, vec3 position, vec3 normal, vec3 eyePos, float coeff) {
  vec3 fragToLightN = normalize(-light.direction);
  vec3 fragToLightReflected = reflect(-fragToLightN, normal);
  vec3 fragToEyeN = normalize(eyePos - position);
  float specular = pow(max(dot(fragToEyeN, fragToLightReflected), 0.0), coeff);
  return light.intensity * light.color * specular;
}
vec3 illuminate(DirectionalLight light, vec3 position, vec3 normal, vec3 eyePos, float specCoeff) {
  vec3 diffuse = illuminateDiffuse(light, normal);
  vec3 specular = illuminateSpecular(light, position, normal, eyePos, specCoeff);
  return diffuse + specular;
}
uniform PointLight pointLight = PointLight(vec3(-2, 1, 2), vec3(1, 1, 1), 2.0f);
uniform DirectionalLight directionalLight = DirectionalLight(vec3(1, 1, 1), vec3(-1, -1, -1), vec3(1, 1, 1), 0.75f);

uniform float u_AmountOfMapNormal = 1.f;
uniform int u_ShadingMode = 0;
uniform bool u_HasSpecular = true;
uniform bool u_UseWhiteAsDiffuse = false;
uniform bool u_IgnoreNormalMap = false;

out vec4 FragColor;

void main() {
  const vec3 objectNormal = normalize(vertexData.objectNormal);
  const vec3 worldNormal = normalize(vertexData.worldNormal);
  const vec3 worldPos = vertexData.worldPosition;
  const vec3 viewPos = u_CameraPosition - worldPos;

  // Calculate TBN in fragment shader
  vec3 p = -viewPos;
  vec2 uv = vertexData.texCoord;
  vec3 N = worldNormal;

  // get edge vectors of the pixel triangle 
  vec3 dp_dx = dFdx(p); 
  vec3 dp_dy = dFdy(p); 
  vec2 duv_dx = dFdx(uv); 
  vec2 duv_dy = dFdy(uv);

  // solve the linear system 
  vec3 dp_dx_perp = cross(N, dp_dx); 
  vec3 dp_dy_perp = cross(dp_dy, N); 
  vec3 T = dp_dy_perp * duv_dx.x + dp_dx_perp * duv_dy.x; 
  vec3 B = dp_dy_perp * duv_dx.y + dp_dx_perp * duv_dy.y;

  // construct a scale-invariant frame 
  float invmax = inversesqrt(max(dot(T,T), dot(B,B))); 
  mat3 tbn = mat3(T * invmax, B * invmax, N);

  vec3 normalMap = texture(secondTex, vertexData.texCoord).rgb;
  //normalMap.y = -normalMap.y;
  vec3 normal = normalize(tbn * (normalMap * 2. - 1.));
  if (u_IgnoreNormalMap)
    normal = worldNormal;
  normal = normalize(mix(worldNormal, normal, u_AmountOfMapNormal));

  // Phong shading
  const float specCoeff = 32.f;
  vec3 diffuseColor = texture(mainTex, vertexData.texCoord).rgb;
  if(u_UseWhiteAsDiffuse)
    diffuseColor = vec3(1);
  vec3 specularColor = u_HasSpecular ? vec3(0.3) : vec3(0.0);
  vec3 directionalDiffuse = illuminateDiffuse(directionalLight, normal);
  vec3 directionalSpecular = illuminateSpecular(directionalLight, worldPos, normal, u_CameraPosition, specCoeff);
  vec3 pointDiffuse = illuminateDiffuse(pointLight, worldPos, normal);
  vec3 pointSpecular = illuminateSpecular(pointLight, worldPos, normal, u_CameraPosition, specCoeff);
  vec3 color = diffuseColor * (directionalDiffuse + pointDiffuse) + specularColor * (directionalSpecular + pointSpecular);

  switch(u_ShadingMode) {
    // Scene
    case 0: {
      FragColor = vec4(color, 1);
    }
    break;
    // diffuseMap
    case 1: {
      FragColor = vec4(diffuseColor, 1);
    }
    break;
    // normalMap
    case 2: {
      FragColor = vec4(normalMap, 1);
    }
    break;
    // vertexNormalInWorld
    case 3: {
      FragColor = vec4(worldNormal * 0.5 + 0.5, 1);
    }
    break;
    // mapNormalInWorld
    case 4: {
      FragColor = vec4(normal * 0.5 + 0.5, 1);
    }
    break;
    // tangent
    case 5: {
      FragColor = vec4(T * invmax * 0.5 + 0.5, 1);
    }
    break;
    // bitangent
    case 6: {
      FragColor = vec4(B * invmax * 0.5 + 0.5, 1);
    }
    break;
  }
}