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
uniform PointLight pointLight = PointLight(vec3(-2, 1, 2), vec3(1, 1, 1), 4.0f);
uniform DirectionalLight directionalLight = DirectionalLight(vec3(1, 1, 1), vec3(-1, -1, -1), vec3(1, 1, 1), 0.75f);

mat3 make_tbn(vec3 N, vec3 p, vec2 uv) { 
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
  return mat3(T * invmax, B * invmax, N);
}

out vec4 FragColor;

void main() {
  const vec3 objectNormal = normalize(vertexData.objectNormal);
  const vec3 worldNormal = normalize(vertexData.worldNormal);

  vec3 wp = vertexData.worldPosition;

  vec3 v = u_CameraPosition - vertexData.worldPosition;
  //v = -v;
  mat3 tbn = make_tbn(worldNormal, -v, vertexData.texCoord);

  vec3 normalMap = texture(secondTex, vertexData.texCoord).rgb * 2.0 - 1.0;
  vec3 normal = normalize(tbn * normalMap);
  //normal = n;

  const float specCoeff = 32.f;

  vec3 diffuseColor = texture(mainTex, vertexData.texCoord).rgb;
  //diffuseColor = vec3(1);
  vec3 specularColor = vec3(1, 1, 1);

  vec3 directionalDiffuse = illuminateDiffuse(directionalLight, normal);
  vec3 directionalSpecular = illuminateSpecular(directionalLight, wp, normal, u_CameraPosition, specCoeff);

  vec3 pointDiffuse = illuminateDiffuse(pointLight, wp, normal);
  vec3 pointSpecular = illuminateSpecular(pointLight, wp, normal, u_CameraPosition, specCoeff);
  //vec3 color = diffuseColor * (directionalDiffuse + pointDiffuse) + specularColor * (directionalSpecular + pointSpecular);
  vec3 color = diffuseColor * (directionalDiffuse + pointDiffuse) + vec3(0.25) * (directionalSpecular + pointSpecular);
  FragColor = vec4(color, 1);

  //FragColor = vec4(normalMap * 0.5 + 0.5, 1);
  //FragColor = vec4(normal * 0.5 + 0.5, 1);
  //FragColor = vec4(b * 0.5 + 0.5, 1);
  //FragColor = vec4(t * 0.5 + 0.5, 1);
}