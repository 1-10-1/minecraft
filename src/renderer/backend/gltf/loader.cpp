#include <mc/renderer/backend/allocator.hpp>
#include <mc/renderer/backend/basisu_transcoder.hpp>
#include <mc/renderer/backend/buffer.hpp>
#include <mc/renderer/backend/gltf/gltfTextures.hpp>
#include <mc/renderer/backend/gltf/loader.hpp>
#include <mc/renderer/backend/image.hpp>
#include <mc/renderer/backend/renderer_backend.hpp>
#include <mc/utils.hpp>

#include <cstring>

#include <glm/gtc/type_ptr.hpp>
#include <vulkan/vulkan_enums.hpp>
#include <vulkan/vulkan_structs.hpp>

namespace renderer::backend
{
    void Model::loadFromFile(std::string filename, float scale)
    {
        tinygltf::Model gltfModel;
        tinygltf::TinyGLTF gltfContext;

        std::string error;
        std::string warning;

        bool binary   = false;
        size_t extpos = filename.rfind('.', filename.length());
        if (extpos != std::string::npos)
        {
            binary = (filename.substr(extpos + 1, filename.length() - extpos) == "glb");
        }

        size_t pos = filename.find_last_of('/');
        if (pos == std::string::npos)
        {
            pos = filename.find_last_of('\\');
        }
        filePath = filename.substr(0, pos);

        // @todo
        gltfContext.SetImageLoader(loadImageDataFunc, nullptr);

        bool fileLoaded = binary
                              ? gltfContext.LoadBinaryFromFile(&gltfModel, &error, &warning, filename.c_str())
                              : gltfContext.LoadASCIIFromFile(&gltfModel, &error, &warning, filename.c_str());

        MC_ASSERT_MSG(fileLoaded, "Could not load gltf file {}", filename);

        LoaderInfo loaderInfo {};
        size_t vertexCount = 0;
        size_t indexCount  = 0;

        extensions = gltfModel.extensionsUsed;
        for (auto& extension : extensions)
        {
            // If this model uses basis universal compressed textures, we need to transcode them
            // So we need to initialize that transcoder once
            if (extension == "KHR_texture_basisu")
            {
                logger::debug("Model uses KHR_texture_basisu, initializing basisu transcoder");
                basist::basisu_transcoder_init();
            }
        }

        loadTextureSamplers(gltfModel);
        loadTextures(gltfModel);
        loadMaterials(gltfModel);

        tinygltf::Scene const& scene =
            gltfModel.scenes[gltfModel.defaultScene > -1 ? gltfModel.defaultScene : 0];

        // Get vertex and index buffer sizes up-front
        for (size_t i = 0; i < scene.nodes.size(); i++)
        {
            getNodeProps(gltfModel.nodes[scene.nodes[i]], gltfModel, vertexCount, indexCount);
        }
        loaderInfo.vertexBuffer = new Vertex[vertexCount];
        loaderInfo.indexBuffer  = new uint32_t[indexCount];

        // TODO: scene handling with no default scene
        for (size_t i = 0; i < scene.nodes.size(); i++)
        {
            tinygltf::Node const node = gltfModel.nodes[scene.nodes[i]];
            loadNode(nullptr, node, scene.nodes[i], gltfModel, loaderInfo, scale);
        }

        if (gltfModel.animations.size() > 0)
        {
            loadAnimations(gltfModel);
        }

        loadSkins(gltfModel);

        for (auto node : linearNodes)
        {
            // Assign skins
            if (node->skinIndex > -1)
            {
                node->skin = skins[node->skinIndex];
            }

            // Initial pose and matrix update
            if (node->mesh)
            {
                node->update();
            }
        }

        primitiveData.reserve(linearNodes.size());
        drawIndirectCommands.reserve(linearNodes.size());

        for (Node* node : nodes)
        {
            preparePrimitiveIndirectData(node);
        }

        primitiveData.shrink_to_fit();
        drawIndirectCommands.shrink_to_fit();

        ScopedCommandBuffer cmdBuf(
            *m_device, m_cmdManager->getTransferCmdPool(), m_device->getTransferQueue(), true);

        // TODO(aether) this is getting trivial
        auto stagingIndirectBuffer = m_bufferManager->create(
            "Draw indirect buffer (staging)",
            drawIndirectCommands.size() * sizeof(decltype(drawIndirectCommands)::value_type),
            vk::BufferUsageFlagBits::eTransferSrc,
            VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
            VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT);

        auto stagingPrimitiveBuffer = m_bufferManager->create(
            "Primitive data buffer (staging)",
            primitiveData.size() * sizeof(decltype(primitiveData)::value_type),
            vk::BufferUsageFlagBits::eTransferSrc,
            VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
            VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT);

        std::memcpy(stagingIndirectBuffer.getMappedData(),
                    drawIndirectCommands.data(),
                    stagingIndirectBuffer.getSize());

        std::memcpy(
            stagingPrimitiveBuffer.getMappedData(), primitiveData.data(), stagingPrimitiveBuffer.getSize());

        drawIndirectBuffer = m_bufferManager->create(
            "Draw indirect buffer",
            drawIndirectCommands.size() * sizeof(decltype(drawIndirectCommands)::value_type),
            vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eIndirectBuffer,
            VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
            VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT);

        primitiveDataBuffer = m_bufferManager->create(
            "Primitive data buffer",
            primitiveData.size() * sizeof(decltype(primitiveData)::value_type),
            vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eShaderDeviceAddress,
            VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
            VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT);

        cmdBuf->copyBuffer(stagingIndirectBuffer,
                           drawIndirectBuffer,
                           vk::BufferCopy().setSize(drawIndirectCommands.size() *
                                                    sizeof(decltype(drawIndirectCommands)::value_type)));
        cmdBuf->copyBuffer(
            stagingPrimitiveBuffer,
            primitiveDataBuffer,
            vk::BufferCopy().setSize(primitiveData.size() * sizeof(decltype(primitiveData)::value_type)));

        primitiveDataBufferAddress =
            m_device->get().getBufferAddress(vk::BufferDeviceAddressInfo().setBuffer(primitiveDataBuffer));

        size_t vertexBufferSize = vertexCount * sizeof(Vertex);
        size_t indexBufferSize  = indexCount * sizeof(uint32_t);

        MC_ASSERT(vertexBufferSize > 0);

        auto vertexStaging = m_bufferManager->create(
            "Vertex staging",
            vertexBufferSize,
            vk::BufferUsageFlagBits::eTransferSrc,
            VMA_MEMORY_USAGE_AUTO,
            VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);

        std::memcpy(vertexStaging.getMappedData(), loaderInfo.vertexBuffer, vertexBufferSize);

        vertices = m_bufferManager->create("Main vertex buffer",
                                           vertexBufferSize,
                                           vk::BufferUsageFlagBits::eTransferDst |
                                               vk::BufferUsageFlagBits::eShaderDeviceAddress,
                                           VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
                                           VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT);

        vertexBufferAddress =
            m_device->get().getBufferAddress(vk::BufferDeviceAddressInfo().setBuffer(vertices));

        cmdBuf->copyBuffer(vertexStaging, vertices, vk::BufferCopy().setSize(vertexBufferSize));

        if (indexBufferSize > 0)
        {
            auto indexStaging = m_bufferManager->create(
                "Index staging",
                indexBufferSize,
                vk::BufferUsageFlagBits::eTransferSrc,
                VMA_MEMORY_USAGE_AUTO,
                VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);

            std::memcpy(indexStaging.getMappedData(), loaderInfo.indexBuffer, indexBufferSize);

            indices = m_bufferManager->create("Main index buffer",
                                              indexBufferSize,
                                              vk::BufferUsageFlagBits::eTransferDst |
                                                  vk::BufferUsageFlagBits::eIndexBuffer,
                                              VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
                                              VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT);

            cmdBuf->copyBuffer(indexStaging, indices, vk::BufferCopy().setSize(indexBufferSize));

            cmdBuf.flush();
        }

        delete[] loaderInfo.vertexBuffer;
        delete[] loaderInfo.indexBuffer;

        getSceneDimensions();

        createMaterialBuffer();
        setupDescriptors();
    }

    Model::~Model()
    {
        for (auto node : nodes)
        {
            delete node;
        }

        for (auto skin : skins)
        {
            delete skin;
        }
    };
}  // namespace renderer::backend
