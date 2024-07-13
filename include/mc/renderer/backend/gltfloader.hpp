#pragma once

#include "buffer.hpp"
#include "descriptor.hpp"
#include "image.hpp"

#include <filesystem>

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
        glm::vec4 color;
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

    struct Texture
    {
        uint32_t imageIndex;
        uint32_t samplerIndex;
    };

    struct GlTFNode
    {
        GlTFNode* parent;
        std::vector<GlTFNode*> children;
        Mesh mesh;
        glm::mat4 transformation;

        ~GlTFNode()
        {
            for (auto& child : children)
            {
                delete child;
            }
        }
    };

    class GlTFScene
    {
    public:
        GlTFScene() = default;

        GlTFScene(Device& device,
                  CommandManager& cmdManager,
                  Allocator& allocator,
                  vk::DescriptorSetLayout materialDescriptorLayout,
                  Image& dummyTexture,
                  vk::Sampler dummySampler,
                  std::filesystem::path path);

        ~GlTFScene()
        {
            for (auto node : m_nodes)
            {
                if (node)
                {
                    delete node;
                }
            }
        };

        GlTFScene(GlTFScene const&) = delete;
        GlTFScene(GlTFScene&&)      = default;

        GlTFScene& operator=(GlTFScene const&) = delete;
        GlTFScene& operator=(GlTFScene&&)      = default;

        void draw(vk::CommandBuffer commandBuffer,
                  vk::Pipeline pipeline,
                  vk::PipelineLayout pipelineLayout,
                  vk::DescriptorSet sceneDataDescriptorSet);

        auto getVertexBufferAddress() const -> size_t { return m_vertexBufferAddress; }

        auto getMaterialBufferAddress() const -> size_t { return m_materialBufferAddress; }

        auto getLastDrawCount() const -> uint32_t { return m_drawCount; }

        auto getLastTriangleCount() const -> uint64_t { return m_triangleCount; }

    private:
        void loadImages(tinygltf::Model& input);

        void loadTextures(tinygltf::Model& input);

        void loadMaterials(tinygltf::Model& input);

        void loadSamplers(tinygltf::Model& input);

        void loadNode(tinygltf::Node const& inputNode,
                      tinygltf::Model const& input,
                      GlTFNode* parent,
                      std::vector<uint32_t>& indexBuffer,
                      std::vector<Vertex>& vertexBuffer);

        void drawNode(vk::CommandBuffer commandBuffer,
                      vk::Pipeline pipeline,
                      vk::PipelineLayout pipelineLayout,
                      vk::DescriptorSet sceneDataDescriptorSet,
                      GlTFNode* node);

        Device* m_device { nullptr };
        CommandManager* m_commandManager { nullptr };
        Allocator* m_allocator { nullptr };

        vk::DescriptorSetLayout m_materialDescriptorLayout { nullptr };
        vk::Sampler m_dummySampler { nullptr };
        Image* m_dummyTexture { nullptr };

        GPUBuffer m_vertexBuffer;
        GPUBuffer m_indexBuffer;

        // TODO(aether) this is, for the moment, immutable
        // TODO(aether) how about we dont keep the host material buffer
        // If we need to change the one on the vram, we copy it over first, then we modify and re-upload just the changed
        // region using BufferCopyRegion or something
        // or instead of the materials we can store more a light-weight array that stores the range of each of the
        // materials on the GPU and whenever material n needs to be modified, we just memcpy it into the
        // (n*sizeof(Material), (n+1)*sizeof(Material)) region of the material buffer on the vram
        // But of course with this, deleting/creating new materials will become more cumbersome
        GPUBuffer m_materialBuffer;
        GPUBuffer m_hostMaterialBuffer;

        size_t m_indexCount;

        std::vector<Image> m_images;
        std::vector<Texture> m_textures;
        std::vector<GlTFNode*> m_nodes;
        std::vector<vk::DescriptorSet> m_materialDescriptors;
        std::vector<vk::raii::Sampler> m_samplers;

        DescriptorAllocator m_descriptorAllocator;

        uint32_t m_drawCount { 0 };
        size_t m_triangleCount { 0 };

        size_t m_vertexBufferAddress { 0 }, m_materialBufferAddress { 0 };
    };
}  // namespace renderer::backend
