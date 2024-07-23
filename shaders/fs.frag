#version 460

#extension GL_GOOGLE_include_directive : require

#include "common.glsl"

layout (set = 1, binding = 0) uniform sampler2D diffuseTexture;
layout (set = 1, binding = 1) uniform sampler2D roughnessMetalnessTexture;
layout (set = 1, binding = 2) uniform sampler2D occlusionTexture;
layout (set = 1, binding = 3) uniform sampler2D emissiveTexture;
layout (set = 1, binding = 4) uniform sampler2D normalTexture;

layout (location = 0) in vec2 vTexcoord0;
layout (location = 1) in vec3 vNormal;
layout (location = 2) in vec4 vTangent;
layout (location = 3) in vec4 vPosition;
layout (location = 4) in vec4 vColor;

layout (location = 0) out vec4 frag_color;

struct AttenuationFactors {
    float quadratic, linear, constant, pad;
};

layout(set = 0, binding = 1) uniform PointLight {
    vec3 position;
    float pad1;
    vec3 color;
    float pad2;
    AttenuationFactors attenuationFactors;
} pointLight;

void main() {
    Material material = sceneData.materialBuffer.materials[materialIndex];

    // frag_color = material.baseColorFactor == vec4(0.0, 0.0, 0.0, 0.0) ? vec4(1.0, 1.0, 1.0, 1.0) : material.baseColorFactor;
    //
    // if ((material.flags & MaterialFeatures_TexcoordVertexAttribute) != 0) {
    //     // A default diffuse texture is supplied no matter what
    //     frag_color *= texture(diffuseTexture, vTexcoord0);
    // } else {
    //     frag_color *= texture(diffuseTexture, gl_FragCoord.xy * 0.0025);
    // }
    //
    frag_color = vec4(1.0, 1.0, 1.0, 1.0);
}

