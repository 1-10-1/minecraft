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

layout(set = 0, binding = 1) uniform PointLight {
    vec3 position;
    float pad1;
    vec3 color;
    float pad2;
    AttenuationFactors attenuationFactors;
} pointLight;

#define PI 3.1415926538

float heaviside( float v ) {
    if ( v > 0.0 ) return 1.0;
    else return 0.0;
}

void main() {
    Material material = sceneData.materialBuffer.materials[materialIndex];

    vec4 diffSample     = texture(textures[nonuniformEXT((materialIndex * 5) + 0)], vTexcoord0);
    vec3 metRoughSample = texture(textures[nonuniformEXT((materialIndex * 5) + 1)], vTexcoord0).rgb;
    vec4 occlSample     = texture(textures[nonuniformEXT((materialIndex * 5) + 2)], vTexcoord0);
    vec4 emisSample     = texture(textures[nonuniformEXT((materialIndex * 5) + 3)], vTexcoord0);
    vec3 normalSample   = texture(textures[nonuniformEXT((materialIndex * 5) + 4)], vTexcoord0).rgb;

    // NOTE(marco): taken from https://community.khronos.org/t/computing-the-tangent-space-in-the-fragment-shader/52861
    vec3 Q1 = dFdx( vPosition.xyz );
    vec3 Q2 = dFdy( vPosition.xyz );
    vec2 st1 = dFdx( vTexcoord0 );
    vec2 st2 = dFdy( vTexcoord0 );

    vec3 T = normalize(  Q1 * st2.t - Q2 * st1.t );
    vec3 B = normalize( -Q1 * st2.s + Q2 * st1.s );

    // the transpose of texture-to-eye space matrix
    mat3 TBN = mat3(
        T,
        B,
        normalize( vNormal )
    );

    vec3 V = normalize( sceneData.cameraPos.xyz - vPosition.xyz );
    vec3 L = normalize( pointLight.position.xyz - vPosition.xyz );
    vec3 N = normalize( vNormal );

    if (material.normalTextureSet != -1) {
        N = normalize( normalSample * 2.0 - 1.0 );
        N = normalize( TBN * N );
    }

    vec3 H = normalize( L + V );

    float roughness = material.roughnessFactor;
    float metalness = material.metallicFactor;

    if (material.physicalDescriptorTextureSet != -1) {
        // Red channel for occlusion value
        // Green channel contains roughness values
        // Blue channel contains metalness
        roughness *= metRoughSample.g;
        metalness *= metRoughSample.b;
    }

    // Could it just be the alpha of the diffuse?
    float alpha = pow(roughness, 2.0);

    float NdotH = dot(N, H);
    float alpha_squared = alpha * alpha;

    float d_denom = ( NdotH * NdotH ) * ( alpha_squared - 1.0 ) + 1.0;

    float distribution = ( alpha_squared * heaviside( NdotH ) )
                       / ( PI * d_denom * d_denom );

    float NdotL = dot(N, L);
    float NdotV = dot(N, V);
    float HdotL = dot(H, L);
    float HdotV = dot(H, V);

    vec4 base_colour = material.baseColorFactor;

    if (material.colorTextureSet != -1) {
        base_colour *= diffSample;
    }

    float visibility = ( heaviside( HdotL ) / ( abs( NdotL ) +
                         sqrt( alpha_squared
                         + ( 1.0 - alpha_squared ) *
                           ( NdotL * NdotL ) ) ) ) *
                           ( heaviside( HdotV ) / ( abs( NdotV ) + sqrt( alpha_squared + ( 1.0 - alpha_squared ) *
                           ( NdotV * NdotV ) ) ) );

    float specular_brdf = visibility * distribution;

    vec3 diffuse_brdf = (1 / PI) * base_colour.rgb;

    // f0 in the formula notation refers to the base colour here
    vec3 conductor_fresnel = specular_brdf * ( base_colour.rgb
                           + ( 1.0 - base_colour.rgb ) * pow( 1.0 - abs( HdotV ), 5 ) );

    float f0 = 0.04; // pow( ( 1 - ior ) / ( 1 + ior ), 2 )
    float fr = f0 + ( 1 - f0 ) * pow(1 - abs( HdotV ), 5 );
    vec3 fresnel_mix = mix( diffuse_brdf, vec3( specular_brdf ), fr );

    vec3 material_colour = mix( fresnel_mix, conductor_fresnel, metalness );

    frag_color = vec4(material_colour, alpha);
}

