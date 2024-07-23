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
    vec3 pos;
    float pad1;
    vec3 normal;
    float pad2;
    vec2 uv0;
    vec2 pad3;
    vec2 uv1;
    vec2 pad4;
    uvec4 joint0;
    vec4 weight0;
    vec4 color;
    vec4 tangent;
};

struct Material {
    vec4 baseColorFactor;
    vec4 emissiveFactor;
    vec4 diffuseFactor;
    vec4 specularFactor;

    float workflow;

    float metallicFactor;
    float emissiveStrength;
    float roughnessFactor;

    int colorTextureSet;
    int normalTextureSet;
    int occlusionTextureSet;
    int emissiveTextureSet;
    int physicalDescriptorTextureSet;

    float alphaMask;
    float alphaMaskCutoff;

    int flags;
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

