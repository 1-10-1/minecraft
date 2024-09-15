#pragma once

#include "../buffer.hpp"
#include "boundingBox.hpp"
#include "constants.hpp"

#include <glm/ext/vector_float2.hpp>
#include <glm/ext/vector_float3.hpp>
#include <glm/ext/vector_float4.hpp>
#include <glm/ext/vector_uint4.hpp>
#include <vulkan/vulkan.hpp>

namespace renderer::backend
{
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

        vk::DrawIndexedIndirectCommand drawCommand;
    };

    struct alignas(16) PrimitiveShaderData
    {
        glm::mat4 matrix;
        uint32_t materialIndex;
    };

    struct Mesh
    {
        Mesh() = default;

        Mesh(ResourceManager<GPUBuffer>& bufferManager, glm::mat4 matrix);
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
            ResourceAccessor<GPUBuffer> buffer;
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
}  // namespace renderer::backend
