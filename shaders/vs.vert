#version 460

#extension GL_GOOGLE_include_directive : require

#include "common.glsl"

layout (location = 0) out vec2 vTexcoord0;
layout (location = 1) out vec2 vTexcoord1;
layout (location = 2) out vec3 vNormal;
layout (location = 3) out vec4 vTangent;
layout (location = 4) out vec4 vPosition;
layout (location = 5) out vec4 vColor;
layout (location = 6) out flat uint vPrimitiveIndex;

void main() {
    Vertex vertex = vertexBuffer.vertices[gl_VertexIndex];
    Primitive primitive = primitiveBuffer.primitives[gl_DrawID];

    gl_Position = scene.viewProj * primitive.matrix * vec4(vertex.pos, 1.0);

    vPosition = primitive.matrix * vec4(vertex.pos, 1.0);
    vNormal = vertex.normal;
    vColor = vertex.color;
    vTexcoord0 = vertex.uv0;
    vTexcoord1 = vertex.uv1;
    vTangent = vertex.tangent;

    vPrimitiveIndex = gl_DrawID;
}
