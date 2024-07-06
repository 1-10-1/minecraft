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
