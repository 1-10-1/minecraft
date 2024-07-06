#version 460

#extension GL_EXT_buffer_reference : require
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_scalar_block_layout : require

#include "uniforms.glsl"

uint MaterialFeatures_ColorTexture     = 1 << 0;
uint MaterialFeatures_NormalTexture    = 1 << 1;
uint MaterialFeatures_RoughnessTexture = 1 << 2;
uint MaterialFeatures_OcclusionTexture = 1 << 3;
uint MaterialFeatures_EmissiveTexture =  1 << 4;
uint MaterialFeatures_TangentVertexAttribute = 1 << 5;
uint MaterialFeatures_TexcoordVertexAttribute = 1 << 6;

struct Vertex {
    vec3 position;
    float uv_x;
    vec3 normal;
    float uv_y;
    vec4 tangent;
};

struct Material {
    vec4 baseColorFactor;

    vec3 emissiveFactor;
    float metallicFactor;

    float roughnessFactor;
    float occlusionFactor;
    uint flags;
    uint pad;
};

layout(buffer_reference, std430) readonly buffer VertexBuffer {
	Vertex vertices[];
};

layout(buffer_reference, std430) readonly buffer MaterialBuffer {
	Material materials[];
};

layout(push_constant) uniform PushConstants
{
    mat4 model;

    // TODO(aether) these can definitely both be constants
    VertexBuffer vertexBuffer;
    MaterialBuffer materialBuffer;

    uint materialIndex;
};

layout (location = 0) out vec2 vTexcoord0;
layout (location = 1) out vec3 vNormal;
layout (location = 2) out vec4 vTangent;
layout (location = 3) out vec4 vPosition;

void main() {
    Vertex vertex = vertexBuffer.vertices[gl_VertexIndex];

    gl_Position = sceneData.viewProj * model * vec4(vertex.position, 1.0);
    vPosition = model * vec4(vertex.position, 1.0);

    vTexcoord0 = vec2(vertex.uv_x, vertex.uv_y);

    // if ( ( flags & MaterialFeatures_TexcoordVertexAttribute ) != 0 ) {
    //     vTexcoord0 = texcoord;
    // }
    // vNormal = mat3( model_inv ) * normal;
    //
    // if ( ( flags & MaterialFeatures_TangentVertexAttribute ) != 0 ) {
    //     vTangent = tangent;
    // }
}
