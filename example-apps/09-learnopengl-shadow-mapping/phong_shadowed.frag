#version 460
#extension GL_ARB_shading_language_include : require

#include "/lib/VertexData.glsl"
#include "/lib/SceneUniforms.glsl"

in VertexData vertexData;
in vec4 FragPosLightSpace;

layout(binding = 0) uniform sampler2D diffuseTexture;
layout(binding = 1) uniform sampler2D shadowMap;

uniform vec3 u_LightPos;
uniform float u_LightIntensity;
uniform vec2 u_ShadowBias; // .x min bias, .y max bias
uniform ivec2 u_ShadowToggles = ivec2(1, 1); // .x shadow=0 outside far plane, .y Percentage Closer Filtering (PCF)

out vec4 FragColor;

float ShadowCalculation(vec4 fragPosLightSpace) {
    const float biasMin = u_ShadowBias.x; // 0.005
    const float biasMax = u_ShadowBias.y; // 0.05
    const bool shouldShadowZeroOutsideFarPlane = bool(u_ShadowToggles.x);
    const bool shouldDoPcf = bool(u_ShadowToggles.y);

    // perform perspective divide
    vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
    // transform to [0,1] range
    projCoords = projCoords * 0.5 + 0.5;
    // get closest depth value from light's perspective (using [0,1] range fragPosLight as coords)
    float closestDepth = texture(shadowMap, projCoords.xy).r; 
    // get depth of current fragment from light's perspective
    float currentDepth = projCoords.z;
    // calculate bias (based on depth map resolution and slope)
    vec3 normal = normalize(vertexData.worldNormal);
    vec3 lightDir = normalize(u_LightPos - vertexData.worldPosition);
    float bias = max(biasMax * (1.0 - dot(normal, lightDir)), biasMin);
    
    // check whether current frag pos is in shadow
    float shadow = 0.0;
    if (shouldDoPcf) {
        vec2 texelSize = 1.0 / textureSize(shadowMap, 0);
        for(int x = -1; x <= 1; ++x)
        {
            for(int y = -1; y <= 1; ++y)
            {
                float pcfDepth = texture(shadowMap, projCoords.xy + vec2(x, y) * texelSize).r; 
                shadow += currentDepth - bias > pcfDepth ? 1.0 : 0.0;        
            }    
        }
        shadow /= 9.0;
    }
    else {
      // shadow = currentDepth > closestDepth  ? 1.0 : 0.0;
      shadow = currentDepth - bias > closestDepth  ? 1.0 : 0.0;
    }

    // keep the shadow at 0.0 when outside the far_plane region of the light's frustum.
    if(shouldShadowZeroOutsideFarPlane && projCoords.z > 1.0)
        shadow = 0.0;

    return shadow;
}

void main() {           
    vec3 color = texture(diffuseTexture, vertexData.texCoord).rgb;
    vec3 normal = normalize(vertexData.worldNormal);
    vec3 lightColor = vec3(0.3);
    // ambient
    vec3 ambient = 0.3 * lightColor;
    // diffuse
    vec3 lightDir = normalize(u_LightPos - vertexData.worldPosition);
    float diff = max(dot(lightDir, normal), 0.0);
    vec3 diffuse = diff * lightColor;
    // specular
    vec3 viewDir = normalize(su.u_CameraPosition - vertexData.worldPosition);
    vec3 reflectDir = reflect(-lightDir, normal);
    float spec = 0.0;
    vec3 halfwayDir = normalize(lightDir + viewDir);  
    spec = pow(max(dot(normal, halfwayDir), 0.0), 64.0);
    vec3 specular = spec * lightColor;    
    // calculate shadow
    float shadow = ShadowCalculation(FragPosLightSpace);                      
    vec3 lighting = (ambient + (1.0 - shadow) * (diffuse + specular) * u_LightIntensity) * color;    
    
    FragColor = vec4(lighting, 1.0);
    //FragColor = gl_FrontFacing ? vec4(1, 0, 0, 1) : vec4(0, 0, 1, 1);
}
