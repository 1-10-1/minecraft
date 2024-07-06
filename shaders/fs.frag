#version 460

#extension GL_GOOGLE_include_directive : require

#include "common.glsl"

uint MaterialFeatures_ColorTexture     = 1 << 0;
uint MaterialFeatures_NormalTexture    = 1 << 1;
uint MaterialFeatures_RoughnessTexture = 1 << 2;
uint MaterialFeatures_OcclusionTexture = 1 << 3;
uint MaterialFeatures_EmissiveTexture =  1 << 4;
uint MaterialFeatures_TangentVertexAttribute = 1 << 5;
uint MaterialFeatures_TexcoordVertexAttribute = 1 << 6;

layout (set = 1, binding = 0) uniform sampler2D diffuseTexture;
layout (set = 1, binding = 1) uniform sampler2D roughnessMetalnessTexture;
layout (set = 1, binding = 2) uniform sampler2D occlusionTexture;
layout (set = 1, binding = 3) uniform sampler2D emissiveTexture;
layout (set = 1, binding = 4) uniform sampler2D normalTexture;

layout (location = 0) in vec2 vTexcoord0;
layout (location = 1) in vec3 vNormal;
layout (location = 2) in vec4 vTangent;
layout (location = 3) in vec4 vPosition;

layout (location = 0) out vec4 frag_color;

void main() {
    Material material = materialBuffer.materials[materialIndex];

    frag_color = texture(diffuseTexture, vTexcoord0) * material.baseColorFactor;

    // frag_color = vec4(texture(diffuseTexture, vTexcoord0).rgb, 1.0);
}

