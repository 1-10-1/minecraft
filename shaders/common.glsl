#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require

struct AttenuationFactors {
    float quadratic, linear, constant;
};

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

layout(set = 0, binding = 0) uniform SceneData {
    mat4 view;
    mat4 proj;
    mat4 viewProj;
    vec4 ambientColor;
    vec3 cameraPos;
    float pad1;
    vec3 sunlightDirection;
} sceneData;

layout(set = 0, binding = 1) uniform PointLight {
    vec3 position;
    float pad1;
    vec3 color;
    float pad2;
    AttenuationFactors attenuationFactors;
    float pad3;
} pointLight;

