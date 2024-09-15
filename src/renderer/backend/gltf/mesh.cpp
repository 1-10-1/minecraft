#include <mc/renderer/backend/gltf/loader.hpp>
#include <mc/renderer/backend/gltf/mesh.hpp>

namespace renderer::backend
{
    Primitive::Primitive(uint32_t firstIndex,
                         uint32_t indexCount,
                         uint32_t vertexCount,
                         uint32_t materialIndex)
        : firstIndex { firstIndex },
          indexCount { indexCount },
          vertexCount { vertexCount },
          materialIndex { materialIndex }
    {
        totalPrims++;

        hasIndices = indexCount > 0;
    };

    void Primitive::setBoundingBox(glm::vec3 min, glm::vec3 max)
    {
        bb.min   = min;
        bb.max   = max;
        bb.valid = true;
    }

    Mesh::Mesh(ResourceManager<GPUBuffer>& bufferManager, glm::mat4 matrix)
    {
        uniformBlock.matrix = matrix;

        // TODO(aether) maybe use REBAR?
        // https://gpuopen-librariesandsdks.github.io/VulkanMemoryAllocator/html/usage_patterns.html#usage_patterns_advanced_data_uploading
        // We'll need to create a separate copying mechanism in GPUBuffer where we check if the memory
        // resides in a device local + host visible memory, and if not, make sure we flush any writes and
        // also copy the staging buffer to the device local one.
        uniformBuffer.buffer = bufferManager.create(
            "Uniform buffer",
            sizeof(uniformBlock),
            vk::BufferUsageFlagBits::eUniformBuffer,
            VMA_MEMORY_USAGE_AUTO,
            VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);

        uniformBuffer.mapped = uniformBuffer.buffer.getMappedData();

        std::memcpy(uniformBuffer.mapped, &uniformBlock, sizeof(UniformBlock));

        uniformBuffer.descriptor = { uniformBuffer.buffer.getVulkanHandle(), 0, sizeof(uniformBlock) };
    };

    void Mesh::setBoundingBox(glm::vec3 min, glm::vec3 max)
    {
        bb.min   = min;
        bb.max   = max;
        bb.valid = true;
    }

    void Model::preparePrimitiveIndirectData(Node* node)
    {
        for (Primitive& primitive : node->mesh->primitives)
        {
            drawIndirectCommands.push_back({
                .indexCount    = primitive.indexCount,
                .instanceCount = 1,
                .firstIndex    = primitive.firstIndex,
                .vertexOffset  = 0,
                .firstInstance = 0,
            });

            triangleCount += primitive.indexCount / 3;

            primitiveData.push_back({
                .matrix        = node->mesh->uniformBlock.matrix * node->matrix,
                .materialIndex = primitive.materialIndex,
            });
        }

        for (Node* n : node->children)
        {
            preparePrimitiveIndirectData(n);
        }
    };
}  // namespace renderer::backend
