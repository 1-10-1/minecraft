#pragma once

#include "../async_loader.hpp"
#include "../buffer.hpp"
#include "../command.hpp"
#include "../descriptor.hpp"
#include "../image.hpp"
#include "animation.hpp"
#include "gltfTextures.hpp"
#include "material.hpp"
#include "mesh.hpp"
#include "node.hpp"

#include <glm/ext/matrix_float4x4.hpp>
#include <glm/ext/quaternion_double.hpp>
#include <glm/ext/vector_float4.hpp>
#include <tiny_gltf.h>

namespace renderer::backend
{
    struct Model
    {
        Model() = default;
        ~Model();

        Model(Device& device,
              CommandManager& cmdManager,
              ResourceManager<Image>& imageManager,
              ResourceManager<GPUBuffer>& bufferManager,
              vk::DescriptorSetLayout materialDescriptorSetLayout,
              vk::ImageView dummyImage,
              vk::Sampler dummySampler,
              AsynchronousLoader& asyncLoader)
            : m_device { &device },
              m_cmdManager { &cmdManager },
              m_imageManager { &imageManager },
              m_bufferManager { &bufferManager },
              m_asyncLoader { &asyncLoader },
              m_materialDescriptorSetLayout { materialDescriptorSetLayout },
              m_dummyImage { dummyImage },
              m_dummySampler { dummySampler }
        {
        }

        Model(Model&&)            = default;
        Model& operator=(Model&&) = default;

        Model(Model const&)            = delete;
        Model& operator=(Model const&) = delete;

        ResourceAccessor<GPUBuffer> indices, vertices, materialBuffer, drawIndirectBuffer,
            primitiveDataBuffer;

        vk::DescriptorSet bindlessMaterialDescriptorSet { nullptr };

        vk::DeviceSize vertexBufferAddress { 0 };
        vk::DeviceSize materialBufferAddress { 0 };
        vk::DeviceSize primitiveDataBufferAddress { 0 };

        glm::mat4 aabb;

        uint64_t triangleCount { 0 };

        std::vector<Node*> nodes;
        std::vector<Node*> linearNodes;

        std::vector<Skin*> skins;

        std::vector<vk::DrawIndexedIndirectCommand> drawIndirectCommands;
        std::vector<PrimitiveShaderData> primitiveData;

        std::vector<GlTFTexture> textures;
        std::vector<TextureSampler> textureSamplers;
        std::vector<Material> materials;
        std::vector<Animation> animations;
        std::vector<std::string> extensions;

        struct Dimensions
        {
            // what
            glm::vec3 min = glm::vec3(std::numeric_limits<float>::max());
            glm::vec3 max = glm::vec3(-std::numeric_limits<float>::max());
        } dimensions;

        struct LoaderInfo
        {
            uint32_t* indexBuffer;
            Vertex* vertexBuffer;
            size_t indexPos  = 0;
            size_t vertexPos = 0;
        };

        std::string filePath;

        void loadNode(Node* parent,
                      tinygltf::Node const& node,
                      uint32_t nodeIndex,
                      tinygltf::Model const& model,
                      LoaderInfo& loaderInfo,
                      float globalscale);

        void getNodeProps(tinygltf::Node const& node,
                          tinygltf::Model const& model,
                          size_t& vertexCount,
                          size_t& indexCount);

        void loadSkins(tinygltf::Model& gltfModel);

        void createMaterialBuffer();

        void loadTextures(tinygltf::Model& gltfModel);

        vk::SamplerAddressMode getVkWrapMode(int32_t wrapMode);

        vk::Filter getVkFilterMode(int32_t filterMode);

        void loadTextureSamplers(tinygltf::Model& gltfModel);

        void loadMaterials(tinygltf::Model& gltfModel);

        void loadAnimations(tinygltf::Model& gltfModel);

        void loadFromFile(std::string filename, float scale = 1.0f);

        void calculateBoundingBox(Node* node, Node* parent);

        void setupDescriptors();

        void getSceneDimensions();

        void updateAnimation(uint32_t index, float time);

        Node* findNode(Node* parent, uint32_t index);

        Node* nodeFromIndex(uint32_t index);

        void preparePrimitiveIndirectData(Node* node);

        static constexpr std::array<std::string_view, 4> const supportedExtensions {
            "KHR_texture_basisu",
            "KHR_materials_pbrSpecularGlossiness",
            "KHR_materials_unlit",
            "KHR_materials_emissive_strength"
        };

    private:
        Device* m_device { nullptr };
        CommandManager* m_cmdManager { nullptr };
        ResourceManager<Image>* m_imageManager { nullptr };
        ResourceManager<GPUBuffer>* m_bufferManager { nullptr };
        AsynchronousLoader* m_asyncLoader { nullptr };

        DescriptorAllocator m_materialDescriptorAllocator {};
        vk::DescriptorSetLayout m_materialDescriptorSetLayout { nullptr };

        vk::ImageView m_dummyImage { nullptr };
        vk::Sampler m_dummySampler { nullptr };
    };
}  // namespace renderer::backend
