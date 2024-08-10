#pragma once

#include "buffer.hpp"
#include "descriptor.hpp"
#include "image.hpp"

#include <filesystem>

#include <glm/ext/matrix_float4x4.hpp>
#include <glm/ext/quaternion_double.hpp>
#include <glm/ext/vector_float4.hpp>
#include <tiny_gltf.h>

namespace renderer::backend
{
    // Changes to this must also be reflected in the shader
    constexpr uint32_t kMaxNumJoints = 128;

    struct Node;

    struct BoundingBox
    {
        BoundingBox() {};

        BoundingBox(glm::vec3 min, glm::vec3 max) : min(min), max(max) {};

        BoundingBox getAABB(glm::mat4 m);

        glm::vec3 min;
        glm::vec3 max;

        bool valid = false;
    };

    struct TextureSampler
    {
        vk::Filter magFilter;
        vk::Filter minFilter;
        vk::SamplerAddressMode addressModeU;
        vk::SamplerAddressMode addressModeV;
        vk::SamplerAddressMode addressModeW;
    };

    struct GlTFTexture
    {
        GlTFTexture() = default;

        ~GlTFTexture() = default;

        GlTFTexture(Device& device,
                    Allocator& allocator,
                    CommandManager& cmdManager,
                    tinygltf::Image& gltfimage,
                    std::filesystem::path path,
                    TextureSampler textureSampler);

        GlTFTexture(GlTFTexture const&)            = delete;
        GlTFTexture& operator=(GlTFTexture const&) = delete;

        GlTFTexture(GlTFTexture&&)            = default;
        GlTFTexture& operator=(GlTFTexture&&) = default;

        BasicImage image {};

        vk::ImageLayout layout {};

        vk::raii::Sampler sampler { nullptr };

    private:
        Device* m_device { nullptr };
        Allocator* m_allocator { nullptr };
        CommandManager* m_commandManager { nullptr };
    };

    struct alignas(16) ShaderMaterial
    {
        glm::vec4 baseColorFactor;
        glm::vec4 emissiveFactor;
        glm::vec4 diffuseFactor;
        glm::vec4 specularFactor;

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

    struct Material
    {
        enum AlphaMode
        {
            ALPHAMODE_OPAQUE,
            ALPHAMODE_MASK,
            ALPHAMODE_BLEND
        };

        AlphaMode alphaMode       = ALPHAMODE_OPAQUE;
        float alphaCutoff         = 1.0f;
        float metallicFactor      = 1.0f;
        float roughnessFactor     = 1.0f;
        glm::vec4 baseColorFactor = glm::vec4(1.0f);
        glm::vec4 emissiveFactor  = glm::vec4(0.0f);
        GlTFTexture* baseColorTexture;
        GlTFTexture* metallicRoughnessTexture;
        GlTFTexture* normalTexture;
        GlTFTexture* occlusionTexture;
        GlTFTexture* emissiveTexture;
        bool doubleSided = false;

        struct TexCoordSets
        {
            uint8_t baseColor          = 0;
            uint8_t metallicRoughness  = 0;
            uint8_t specularGlossiness = 0;
            uint8_t normal             = 0;
            uint8_t occlusion          = 0;
            uint8_t emissive           = 0;
        } texCoordSets;

        struct Extension
        {
            GlTFTexture* specularGlossinessTexture;
            GlTFTexture* diffuseTexture;
            glm::vec4 diffuseFactor  = glm::vec4(1.0f);
            glm::vec3 specularFactor = glm::vec3(0.0f);
        } extension;

        // TODO(aether) this is weird, make it an enum instead
        struct PbrWorkflows
        {
            bool metallicRoughness  = true;
            bool specularGlossiness = false;
        } pbrWorkflows;

        int index              = 0;
        bool unlit             = false;
        float emissiveStrength = 1.0f;
    };

    struct alignas(16) Vertex
    {
        // TODO(aether) these manual pads might be unnecessary
        glm::vec3 pos;
        float pad1;
        glm::vec3 normal;
        float pad2;
        glm::vec2 uv0;
        glm::vec2 pad3;
        glm::vec2 uv1;
        glm::vec2 pad4;
        glm::uvec4 joint0;
        glm::vec4 weight0;
        glm::vec4 color;
        glm::vec4 tangent;
    };

    struct Primitive
    {
        Primitive(uint32_t firstIndex, uint32_t indexCount, uint32_t vertexCount, uint32_t materialIndex);

        void setBoundingBox(glm::vec3 min, glm::vec3 max);

        uint32_t firstIndex;
        uint32_t indexCount;
        uint32_t vertexCount;

        uint32_t materialIndex;

        bool hasIndices;

        BoundingBox bb;

        inline static uint64_t totalPrims = 0;
    };

    struct Mesh
    {
        Mesh() = default;

        Mesh(Allocator& allocator, glm::mat4 matrix);
        ~Mesh() = default;

        Mesh(Mesh const&)            = delete;
        Mesh& operator=(Mesh const&) = delete;

        Mesh(Mesh&&)            = default;
        Mesh& operator=(Mesh&&) = default;

        void setBoundingBox(glm::vec3 min, glm::vec3 max);

        Allocator* allocator;

        std::vector<Primitive> primitives;

        BoundingBox bb;
        BoundingBox aabb;

        struct UniformBuffer
        {
            GPUBuffer buffer;
            VkDescriptorBufferInfo descriptor;
            VkDescriptorSet descriptorSet;
            void* mapped;
        } uniformBuffer;

        struct UniformBlock
        {
            glm::mat4 matrix;
            glm::mat4 jointMatrix[kMaxNumJoints] {};
            uint32_t jointcount { 0 };
        } uniformBlock;
    };

    struct Skin
    {
        std::string name;
        Node* skeletonRoot = nullptr;

        std::vector<glm::mat4> inverseBindMatrices;
        std::vector<Node*> joints;
    };

    struct Node
    {
        void update();

        ~Node()
        {
            for (auto& children : children)
            {
                delete children;
            }
        };

        glm::mat4 localMatrix();
        glm::mat4 getMatrix();

        std::string name;

        Node* parent;
        std::vector<Node*> children;

        uint32_t index;
        glm::mat4 matrix;

        std::unique_ptr<Mesh> mesh;

        Skin* skin;
        int32_t skinIndex = -1;

        glm::vec3 translation {};
        glm::vec3 scale { 1.0f };
        glm::dquat rotation {};

        BoundingBox bvh;
        BoundingBox aabb;

        glm::mat4 cachedLocalMatrix { glm::mat4(1.0f) };
        glm::mat4 cachedMatrix { glm::mat4(1.0f) };

        bool useCachedMatrix { false };
    };

    struct AnimationChannel
    {
        enum PathType
        {
            TRANSLATION,
            ROTATION,
            SCALE
        };

        PathType path;
        Node* node;
        uint32_t samplerIndex;
    };

    struct AnimationSampler
    {
        enum InterpolationType
        {
            LINEAR,
            STEP,
            CUBICSPLINE
        };

        InterpolationType interpolation;
        std::vector<float> inputs;
        std::vector<glm::vec4> outputsVec4;
        std::vector<float> outputs;
        glm::vec4 cubicSplineInterpolation(size_t index, float time, uint32_t stride);
        void translate(size_t index, float time, Node* node);
        void scale(size_t index, float time, Node* node);
        void rotate(size_t index, float time, Node* node);
    };

    struct Animation
    {
        std::string name;
        std::vector<AnimationSampler> samplers;
        std::vector<AnimationChannel> channels;
        float start = std::numeric_limits<float>::max();
        float end   = std::numeric_limits<float>::min();
    };

    struct Model
    {
        Model() = default;
        ~Model();

        Model(Device& device,
              Allocator& allocator,
              CommandManager& cmdManager,
              vk::DescriptorSetLayout materialDescriptorSetLayout,
              vk::ImageView dummyImage,
              vk::Sampler dummySampler)
            : device { &device },
              allocator { &allocator },
              cmdManager { &cmdManager },
              m_materialDescriptorSetLayout { materialDescriptorSetLayout },
              m_dummyImage { dummyImage },
              m_dummySampler { dummySampler }
        {
        }

        Model(Model&&)            = default;
        Model& operator=(Model&&) = default;

        Model(Model const&)            = delete;
        Model& operator=(Model const&) = delete;

        GPUBuffer indices;
        GPUBuffer vertices;
        GPUBuffer materialBuffer;

        vk::DescriptorSet bindlessMaterialDescriptorSet { nullptr };

        vk::DeviceSize vertexBufferAddress { 0 };
        vk::DeviceSize materialBufferAddress { 0 };

        glm::mat4 aabb;

        // make nodes a vector of unique ptrs maybe?
        std::vector<Node*> nodes;
        std::vector<Node*> linearNodes;

        std::vector<Skin*> skins;

        std::vector<GlTFTexture> textures;
        std::vector<TextureSampler> textureSamplers;
        std::vector<Material> materials;
        std::vector<Animation> animations;
        std::vector<std::string> extensions;

        struct Dimensions
        {
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

        // These are only called for the skybox, not the scene (for some reason, figure it out)
        void drawNode(Node* node, vk::CommandBuffer commandBuffer);

        void draw(vk::CommandBuffer commandBuffer);

        void calculateBoundingBox(Node* node, Node* parent);

        void setupDescriptors();

        void getSceneDimensions();

        void updateAnimation(uint32_t index, float time);

        Node* findNode(Node* parent, uint32_t index);

        Node* nodeFromIndex(uint32_t index);

        static constexpr std::array<std::string_view, 4> const supportedExtensions {
            "KHR_texture_basisu",
            "KHR_materials_pbrSpecularGlossiness",
            "KHR_materials_unlit",
            "KHR_materials_emissive_strength"
        };

    private:
        Device* device { nullptr };
        Allocator* allocator { nullptr };
        CommandManager* cmdManager { nullptr };

        DescriptorAllocator m_materialDescriptorAllocator {};
        vk::DescriptorSetLayout m_materialDescriptorSetLayout { nullptr };

        vk::ImageView m_dummyImage { nullptr };
        vk::Sampler m_dummySampler { nullptr };
    };
}  // namespace renderer::backend
