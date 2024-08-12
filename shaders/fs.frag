#version 460

#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_nonuniform_qualifier : enable

#include "common.glsl"

layout (set = 1, binding = 0) uniform sampler2D textures[];

layout (location = 0) in vec2 vTexcoord0;
layout (location = 1) in vec2 vTexcoord1;
layout (location = 2) in vec3 vNormal;
layout (location = 3) in vec4 vTangent;
layout (location = 4) in vec4 vPosition;
layout (location = 5) in vec4 vColor;

layout (location = 0) out vec4 frag_color;

struct AttenuationFactors {
    float quadratic, linear, constant, pad;
};

#define PI 3.1415926538

float heaviside( float v ) {
    if ( v > 0.0 ) return 1.0;
    else return 0.0;
}

void main() {
    Material material = scene.materialBuffer.materials[materialIndex];

    vec4 diffSample     = texture(textures[nonuniformEXT((materialIndex * 5) + 0)], vTexcoord0);
    vec3 metRoughSample = texture(textures[nonuniformEXT((materialIndex * 5) + 1)], vTexcoord0).rgb;
    vec4 occlSample     = texture(textures[nonuniformEXT((materialIndex * 5) + 2)], vTexcoord0);
    vec4 emisSample     = texture(textures[nonuniformEXT((materialIndex * 5) + 3)], vTexcoord0);
    vec3 normalSample   = texture(textures[nonuniformEXT((materialIndex * 5) + 4)], vTexcoord0).rgb;

    frag_color = diffSample;
    
    if (material.colorTextureSet != -1) {
        frag_color *= material.baseColorFactor;
    }
}

