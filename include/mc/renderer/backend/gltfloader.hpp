#pragma once

#include "buffer.hpp"
#include "descriptor.hpp"
#include "image.hpp"

#include <glm/ext/matrix_float4x4.hpp>
#include <glm/ext/vector_float4.hpp>
#include <tiny_gltf.h>

namespace renderer::backend
{
    enum class MaterialFeatures : uint32_t
    {
        ColorTexture            = 1 << 0,
        NormalTexture           = 1 << 1,
        RoughnessTexture        = 1 << 2,
        OcclusionTexture        = 1 << 3,
        EmissiveTexture         = 1 << 4,
        TangentVertexAttribute  = 1 << 5,
        TexcoordVertexAttribute = 1 << 6,
    };

    struct alignas(16) Material
    {
        glm::vec4 baseColorFactor;

        glm::vec3 emissiveFactor;
        float metallicFactor;

        float roughnessFactor;
        float occlusionFactor;
        uint32_t flags;
        uint32_t pad;
    };

    struct alignas(16) Vertex
    {
        glm::vec3 position;
        float uv_x;
        glm::vec3 normal;
        float uv_y;
        glm::vec4 tangent;
    };

    struct Primitive
    {
        uint32_t firstIndex;
        uint32_t indexCount;
        int32_t materialIndex;
    };

    struct Mesh
    {
        std::vector<Primitive> primitives;
    };

    struct GltfNode
    {
        GltfNode* parent;
        std::vector<GltfNode*> children;
        Mesh mesh;
        glm::mat4 transformation;

        ~GltfNode()
        {
            for (auto& child : children)
            {
                delete child;
            }
        }
    };

    struct GltfImage
    {
        Texture texture;
    };

    struct GltfTexture
    {
        uint32_t imageIndex;
        uint32_t samplerIndex;
    };

    struct MaterialRenderInfo
    {
        uint32_t baseColorTextureIndex;
        uint32_t normalTextureIndex;
        uint32_t roughnessTextureIndex;
        uint32_t occlusionTextureIndex;
        uint32_t emissiveTextureIndex;

        vk::DescriptorSet descriptorSet;
    };

    struct SceneResources
    {
        GPUBuffer vertexBuffer;
        GPUBuffer indexBuffer;

        // materialBuffer is a dedicated buffer on the GPU
        // hostMaterialBuffer is the staging buffer that gets copied to the one on the GPU whenever a change is requested
        //
        // TODO(aether) this is, for the moment, immutable
        //
        // TODO(aether) how about we dont keep the host material buffer
        // If we need to change the one on the vram, we copy it over first, then we modify and re-upload just the changed
        // region using BufferCopyRegion or something
        // or instead of the materials we can store more a light-weight array that stores the range of each of the
        // materials on the GPU and whenever material n needs to be modified, we just memcpy it into the
        // (n*sizeof(Material), (n+1)*sizeof(Material)) region of the material buffer on the vram
        // But of course with this, deleting/creating new materials will become more cumbersome
        GPUBuffer materialBuffer;
        GPUBuffer hostMaterialBuffer;

        size_t indexCount;

        std::vector<GltfImage> images;
        std::vector<GltfTexture> textures;
        std::vector<GltfNode*> nodes;
        std::vector<vk::DescriptorSet> materialDescriptors;
        std::vector<vk::raii::Sampler> samplers;

        DescriptorAllocator descriptorAllocator;

        ~SceneResources()
        {
            for (auto node : nodes)
            {
                delete node;
            }
        };
    };
}  // namespace renderer::backend
