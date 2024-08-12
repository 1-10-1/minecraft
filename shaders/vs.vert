#version 460

#extension GL_GOOGLE_include_directive : require

#include "common.glsl"

layout (location = 0) out vec2 vTexcoord0;
layout (location = 1) out vec2 vTexcoord1;
layout (location = 2) out vec3 vNormal;
layout (location = 3) out vec4 vTangent;
layout (location = 4) out vec4 vPosition;
layout (location = 5) out vec4 vColor;

void main() {
    Vertex vertex = scene.vertexBuffer.vertices[gl_VertexIndex];
    Material material = scene.materialBuffer.materials[materialIndex];

    gl_Position = scene.viewProj * model * vec4(vertex.pos, 1.0);

    vPosition = model * vec4(vertex.pos, 1.0);
    vNormal = vertex.normal;
    vColor = vertex.color;
    vTexcoord0 = vertex.uv0;
    vTexcoord1 = vertex.uv1;
    vTangent = vertex.tangent;

    // CAREFUL! We used to generate normals if they weren't found before, but no longer
    // vNormal = mat3(model_inv) * normal;
}
