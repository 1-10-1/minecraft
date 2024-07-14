#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require

uint MaterialFeatures_ColorTexture     = 1 << 0;
uint MaterialFeatures_NormalTexture    = 1 << 1;
uint MaterialFeatures_RoughnessTexture = 1 << 2;
uint MaterialFeatures_OcclusionTexture = 1 << 3;
uint MaterialFeatures_EmissiveTexture =  1 << 4;
uint MaterialFeatures_TangentVertexAttribute = 1 << 5;
uint MaterialFeatures_TexcoordVertexAttribute = 1 << 6;

layout(push_constant) uniform PushConstants
{
    mat4 model;
    uint materialIndex;
};

struct Vertex {
    vec3 position;
    float uv_x;
    vec3 normal;
    float uv_y;
    vec4 tangent;
    vec4 color;
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

layout(set = 0, binding = 0) uniform SceneData {
    mat4 view;
    mat4 proj;
    mat4 viewProj;
    vec4 ambientColor;
    vec3 cameraPos;
    float screenWidth;
    vec3 sunlightDirection;
    float screenHeight;
    VertexBuffer vertexBuffer;
    MaterialBuffer materialBuffer;
} sceneData;

