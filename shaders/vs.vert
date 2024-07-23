#version 460

#extension GL_GOOGLE_include_directive : require

#include "common.glsl"

layout (location = 0) out vec2 vTexcoord0;
layout (location = 1) out vec3 vNormal;
layout (location = 2) out vec4 vTangent;
layout (location = 3) out vec4 vPosition;
layout (location = 4) out vec4 vColor;

void main() {
    Vertex vertex = sceneData.vertexBuffer.vertices[gl_VertexIndex];
    Material material = sceneData.materialBuffer.materials[materialIndex];

    gl_Position = sceneData.viewProj * model * vec4(vertex.pos, 1.0);
    vPosition = model * vec4(vertex.pos, 1.0);

    if ((material.flags & MaterialFeatures_TexcoordVertexAttribute) != 0) {
        vTexcoord0 = vertex.uv0;
    }

    if ((material.flags & MaterialFeatures_TangentVertexAttribute) != 0) {
        vTangent = vertex.tangent;
    }

    vColor = vertex.color;

    // CAREFUL! We used to generate normals if they weren't found before, but no longer
    // vNormal = mat3(model_inv) * normal;
}
