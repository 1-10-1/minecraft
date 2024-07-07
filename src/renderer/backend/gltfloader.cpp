#include <cstring>
#include <mc/renderer/backend/allocator.hpp>
#include <mc/renderer/backend/gltfloader.hpp>
#include <mc/renderer/backend/renderer_backend.hpp>

#include <filesystem>

#include <glm/gtc/type_ptr.hpp>
#include <vulkan/vulkan_structs.hpp>

namespace
{
    vk::SamplerAddressMode getVkWrapMode(int32_t wrapMode)
    {
        switch (wrapMode)
        {
            case -1:
            case 10497:
                return vk::SamplerAddressMode::eRepeat;
            case 33071:
                return vk::SamplerAddressMode::eClampToEdge;
            case 33648:
                return vk::SamplerAddressMode::eMirroredRepeat;
        }

        MC_ASSERT_MSG(false, "Unknown wrap mode: ", wrapMode);
    }

    vk::Filter getVkFilterMode(int32_t filterMode)
    {
        switch (filterMode)
        {
            case -1:
            case 9728:
                return vk::Filter::eNearest;
            case 9729:
                return vk::Filter::eNearest;
            case 9984:
                return vk::Filter::eNearest;
            case 9985:
                return vk::Filter::eNearest;
            case 9986:
                return vk::Filter::eLinear;
            case 9987:
                return vk::Filter::eLinear;
        }

        MC_ASSERT_MSG(false, "Unknown filter mode: ", filterMode);
    }
}  // namespace

namespace renderer::backend
{
    namespace fs = std::filesystem;

    void RendererBackend::processGltf()
    {
#if 0
        fs::path gltfDir  = "../../khrSampleModels/2.0/Cube/glTF";
        fs::path prevPath = fs::current_path();
        fs::current_path(gltfDir);
        fs::path path = fs::current_path() / "Cube.gltf";
#elif 1
        fs::path gltfDir  = "../../khrSampleModels/2.0/DragonAttenuation/glTF";
        fs::path prevPath = fs::current_path();
        fs::current_path(gltfDir);
        fs::path path = fs::current_path() / "DragonAttenuation.gltf";

#else
        fs::path gltfDir  = "../../khrSampleModels/2.0/Sponza/glTF";
        fs::path prevPath = fs::current_path();
        fs::current_path(gltfDir);
        fs::path path = fs::current_path() / "Sponza.gltf";
#endif

        MC_ASSERT_MSG(fs::exists(path), "glTF file path does not exist: {}", path.string());

        tinygltf::Model glTFInput;
        tinygltf::TinyGLTF gltfContext;
        std::string error, warning;

        MC_ASSERT(gltfContext.LoadASCIIFromFile(&glTFInput, &error, &warning, path));

        // TODO(aether) maybe you could half this?
        std::vector<uint32_t> indexBuffer;
        std::vector<Vertex> vertexBuffer;

        loadImages(glTFInput);
        loadTextures(glTFInput);
        loadSamplers(glTFInput);
        loadMaterials(glTFInput);

        tinygltf::Scene const& scene = glTFInput.scenes[0];

        for (size_t i = 0; i < scene.nodes.size(); i++)
        {
            tinygltf::Node const node = glTFInput.nodes[scene.nodes[i]];
            loadNode(node, glTFInput, nullptr, indexBuffer, vertexBuffer);
        }

        // We delayed this so that loadNode can set the vertex attribute material flags
        // such as "TangentVertexAttribute", "TexcoordVertexAttribute", etc.
        ScopedCommandBuffer(
            m_device, m_commandManager.getTransferCmdPool(), m_device.getTransferQueue(), true)
            ->copyBuffer(m_sceneResources.hostMaterialBuffer,
                         m_sceneResources.materialBuffer,
                         vk::BufferCopy().setSize(m_sceneResources.materialBuffer.getSize()));

        size_t vertexBufferSize     = vertexBuffer.size() * sizeof(Vertex);
        size_t indexBufferSize      = indexBuffer.size() * sizeof(uint32_t);
        m_sceneResources.indexCount = static_cast<uint32_t>(indexBuffer.size());

        GPUBuffer vertexStaging(m_allocator,
                                vertexBufferSize,
                                vk::BufferUsageFlagBits::eTransferSrc,
                                VMA_MEMORY_USAGE_AUTO,
                                VMA_ALLOCATION_CREATE_MAPPED_BIT |
                                    VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);

        GPUBuffer indexStaging(
            m_allocator,
            vertexBufferSize,
            vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eShaderDeviceAddress,
            VMA_MEMORY_USAGE_AUTO,
            VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);

        // TODO(aether) maybe just modify the mapped region instead of creating two copies?
        std::memcpy(indexStaging.getMappedData(), indexBuffer.data(), indexBufferSize);
        std::memcpy(vertexStaging.getMappedData(), vertexBuffer.data(), vertexBufferSize);

        m_sceneResources.vertexBuffer =
            GPUBuffer(m_allocator,
                      vertexBufferSize,
                      vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eShaderDeviceAddress,
                      VMA_MEMORY_USAGE_AUTO,
                      VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT);

        m_sceneResources.indexBuffer =
            GPUBuffer(m_allocator,
                      indexBufferSize,
                      vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eIndexBuffer,
                      VMA_MEMORY_USAGE_AUTO,
                      VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT);

        ScopedCommandBuffer cmdBuf(
            m_device, m_commandManager.getTransferCmdPool(), m_device.getTransferQueue());

        cmdBuf->copyBuffer(
            indexStaging, m_sceneResources.indexBuffer, vk::BufferCopy().setSize(indexBufferSize));

        cmdBuf->copyBuffer(
            vertexStaging, m_sceneResources.vertexBuffer, vk::BufferCopy().setSize(vertexBufferSize));
    }

    void RendererBackend::loadImages(tinygltf::Model& input)
    {
        // Images can be stored inside the glTF (which is the case for the sample
        // model), so instead of directly loading them from disk, we fetch them from
        // the glTF loader and upload the buffers
        m_sceneResources.images.resize(input.images.size());

        for (size_t i = 0; i < input.images.size(); i++)
        {
            tinygltf::Image& glTFImage = input.images[i];
            // Get the image data from the glTF loader
            unsigned char* buffer   = nullptr;
            VkDeviceSize bufferSize = 0;
            bool deleteBuffer       = false;
            // We convert RGB-only images to RGBA, as most devices don't support
            // RGB-formats in Vulkan
            if (glTFImage.component == 3)
            {
                bufferSize          = glTFImage.width * glTFImage.height * 4;
                buffer              = new unsigned char[bufferSize];
                unsigned char* rgba = buffer;
                unsigned char* rgb  = &glTFImage.image[0];
                for (size_t i = 0; i < glTFImage.width * glTFImage.height; ++i)
                {
                    memcpy(rgba, rgb, sizeof(unsigned char) * 3);
                    rgba += 4;
                    rgb += 3;
                }
                deleteBuffer = true;
            }
            else
            {
                buffer     = &glTFImage.image[0];
                bufferSize = glTFImage.image.size();
            }

            // Load texture from image buffer
            m_sceneResources.images[i].texture =
                Texture(m_device,
                        m_allocator,
                        m_commandManager,
                        vk::Extent2D { static_cast<uint32_t>(glTFImage.width),
                                       static_cast<uint32_t>(glTFImage.height) },
                        buffer,
                        bufferSize);

            if (deleteBuffer)
            {
                delete[] buffer;
            }
        }
    };

    void RendererBackend::loadSamplers(tinygltf::Model& input)
    {
        m_sceneResources.samplers.reserve(input.samplers.size());

        for (tinygltf::Sampler& sampler : input.samplers)
        {
            m_sceneResources.samplers.push_back(
                m_device->createSampler({
                    .magFilter               = getVkFilterMode(sampler.magFilter),
                    .minFilter               = getVkFilterMode(sampler.minFilter),
                    .mipmapMode              = vk::SamplerMipmapMode::eLinear,
                    .addressModeU            = getVkWrapMode(sampler.wrapS),
                    .addressModeV            = getVkWrapMode(sampler.wrapT),
                    .addressModeW            = getVkWrapMode(sampler.wrapT),
                    .mipLodBias              = 0.0f,
                    .anisotropyEnable        = false,
                    .maxAnisotropy           = m_device.getDeviceProperties().limits.maxSamplerAnisotropy,
                    .compareEnable           = false,
                    .minLod                  = 0.0f,
                    .maxLod                  = 1,
                    .borderColor             = vk::BorderColor::eIntOpaqueBlack,
                    .unnormalizedCoordinates = false,
                }) >>
                ResultChecker());
        }
    }

    void RendererBackend::loadTextures(tinygltf::Model& input)
    {
        m_sceneResources.textures.resize(input.textures.size());

        for (size_t i = 0; i < input.textures.size(); i++)
        {
            m_sceneResources.textures[i].imageIndex   = input.textures[i].source;
            m_sceneResources.textures[i].samplerIndex = input.textures[i].sampler;
        }
    };

    void RendererBackend::loadMaterials(tinygltf::Model& input)
    {
        m_sceneResources.materialDescriptors.resize(input.materials.size());

        std::vector<DescriptorAllocator::PoolSizeRatio> sizes = {
            { vk::DescriptorType::eCombinedImageSampler, 5 },
        };

        m_sceneResources.descriptorAllocator = DescriptorAllocator(m_device, input.materials.size(), sizes);

        m_sceneResources.hostMaterialBuffer = GPUBuffer(
            m_allocator,
            sizeof(Material) * input.materials.size(),
            vk::BufferUsageFlagBits::eTransferSrc,
            VMA_MEMORY_USAGE_AUTO,
            VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT);

        m_sceneResources.materialBuffer =
            GPUBuffer(m_allocator,
                      m_sceneResources.hostMaterialBuffer.getSize(),
                      vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eShaderDeviceAddress,
                      VMA_MEMORY_USAGE_AUTO,
                      VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT);

        std::memset(m_sceneResources.hostMaterialBuffer.getMappedData(),
                    0,
                    m_sceneResources.hostMaterialBuffer.getSize());

        std::vector<int> b {};

        for (size_t i = 0; i < input.materials.size(); i++)
        {
            tinygltf::Material glTFMaterial = input.materials[i];

            Material& material =
                reinterpret_cast<Material*>(m_sceneResources.hostMaterialBuffer.getMappedData())[i];

            vk::DescriptorSet descriptorSet =
                m_sceneResources.descriptorAllocator.allocate(m_device, m_materialDescriptorLayout);

            DescriptorWriter writer;

            if (glTFMaterial.values.find("baseColorFactor") != glTFMaterial.values.end())
            {
                material.baseColorFactor =
                    glm::make_vec4(glTFMaterial.values["baseColorFactor"].ColorFactor().data());
            }
            else
            {
                material.baseColorFactor = { 1.0, 1.0, 1.0, 1.0 };
            }

            for (auto const& [binding, pair] : vi::enumerate(std::array {
                     std::pair { "baseColorTexture",         MaterialFeatures::ColorTexture     },
                     std::pair { "metallicRoughnessTexture", MaterialFeatures::RoughnessTexture },
                     std::pair { "occlusionTexture",         MaterialFeatures::OcclusionTexture },
                     std::pair { "emissiveTexture",          MaterialFeatures::EmissiveTexture  },
                     std::pair { "normalTexture",            MaterialFeatures::NormalTexture    },
            }))
            {
                char const* name                          = pair.first;
                [[maybe_unused]] MaterialFeatures feature = pair.second;

                if (glTFMaterial.values.find(name) != glTFMaterial.values.end())
                {
                    writer.write_image(
                        binding,
                        m_sceneResources.images[glTFMaterial.values[name].TextureIndex()]
                            .texture.getImageView(),
                        m_sceneResources
                            .samplers[m_sceneResources
                                          .textures[glTFMaterial.additionalValues[name].TextureIndex()]
                                          .samplerIndex],
                        vk::ImageLayout::eShaderReadOnlyOptimal,
                        vk::DescriptorType::eCombinedImageSampler);

                    material.flags |= std::to_underlying(feature);
                }
                else if (glTFMaterial.additionalValues.find(name) != glTFMaterial.additionalValues.end())
                {
                    writer.write_image(
                        binding,
                        m_sceneResources.images[glTFMaterial.additionalValues[name].TextureIndex()]
                            .texture.getImageView(),
                        m_sceneResources
                            .samplers[m_sceneResources
                                          .textures[glTFMaterial.additionalValues[name].TextureIndex()]
                                          .samplerIndex],
                        vk::ImageLayout::eShaderReadOnlyOptimal,
                        vk::DescriptorType::eCombinedImageSampler);

                    material.flags |= std::to_underlying(feature);
                }
                else
                {
                    writer.write_image(binding,
                                       m_dummyTexture.getImageView(),
                                       m_dummySampler,
                                       vk::ImageLayout::eShaderReadOnlyOptimal,
                                       vk::DescriptorType::eCombinedImageSampler);
                }
            }

            writer.update_set(m_device, descriptorSet);

            m_sceneResources.materialDescriptors[i] = descriptorSet;
        }
    };

    void RendererBackend::loadNode(tinygltf::Node const& inputNode,
                                   tinygltf::Model const& input,
                                   GltfNode* parent,
                                   std::vector<uint32_t>& indexBuffer,
                                   std::vector<Vertex>& vertexBuffer)
    {
        GltfNode* node       = new GltfNode {};
        node->transformation = glm::mat4(1.0f);
        node->parent         = parent;

        // Get the local node matrix
        // It's either made up from translation, rotation, scale or a 4x4 matrix
        if (inputNode.translation.size() == 3)
        {
            node->transformation =
                glm::translate(node->transformation, glm::vec3(glm::make_vec3(inputNode.translation.data())));
        }

        if (inputNode.rotation.size() == 4)
        {
            glm::quat q = glm::make_quat(inputNode.rotation.data());
            node->transformation *= glm::mat4(q);
        }
        if (inputNode.scale.size() == 3)
        {
            node->transformation =
                glm::scale(node->transformation, glm::vec3(glm::make_vec3(inputNode.scale.data())));
        }

        if (inputNode.matrix.size() == 16)
        {
            node->transformation = glm::make_mat4x4(inputNode.matrix.data());
        };

        // Load node's children
        if (inputNode.children.size() > 0)
        {
            for (size_t i = 0; i < inputNode.children.size(); i++)
            {
                loadNode(input.nodes[inputNode.children[i]], input, node, indexBuffer, vertexBuffer);
            }
        }

        // If the node contains mesh data, we load vertices and indices from the
        // buffers In glTF this is done via accessors and buffer views
        if (inputNode.mesh > -1)
        {
            tinygltf::Mesh const mesh = input.meshes[inputNode.mesh];
            // Iterate through all primitives of this node's mesh
            for (size_t i = 0; i < mesh.primitives.size(); i++)
            {
                tinygltf::Primitive const& glTFPrimitive = mesh.primitives[i];

                Material& material = static_cast<Material*>(
                    m_sceneResources.hostMaterialBuffer.getMappedData())[glTFPrimitive.material];

                uint32_t firstIndex  = static_cast<uint32_t>(indexBuffer.size());
                uint32_t vertexStart = static_cast<uint32_t>(vertexBuffer.size());
                uint32_t indexCount  = 0;
                // Vertices
                {
                    float const* positionBuffer  = nullptr;
                    float const* normalsBuffer   = nullptr;
                    float const* texCoordsBuffer = nullptr;
                    float const* tangentsBuffer  = nullptr;
                    size_t vertexCount           = 0;

                    // Get buffer data for vertex positions
                    if (glTFPrimitive.attributes.find("POSITION") != glTFPrimitive.attributes.end())
                    {
                        tinygltf::Accessor const& accessor =
                            input.accessors[glTFPrimitive.attributes.find("POSITION")->second];
                        tinygltf::BufferView const& view = input.bufferViews[accessor.bufferView];
                        positionBuffer                   = reinterpret_cast<float const*>(
                            &(input.buffers[view.buffer].data[accessor.byteOffset + view.byteOffset]));
                        vertexCount = accessor.count;
                    }

                    // Get buffer data for vertex normals
                    if (glTFPrimitive.attributes.find("NORMAL") != glTFPrimitive.attributes.end())
                    {
                        tinygltf::Accessor const& accessor =
                            input.accessors[glTFPrimitive.attributes.find("NORMAL")->second];
                        tinygltf::BufferView const& view = input.bufferViews[accessor.bufferView];
                        normalsBuffer                    = reinterpret_cast<float const*>(
                            &(input.buffers[view.buffer].data[accessor.byteOffset + view.byteOffset]));
                    }

                    // Get buffer data for vertex texture coordinates
                    // glTF supports multiple sets, we only load the first one
                    if (glTFPrimitive.attributes.find("TEXCOORD_0") != glTFPrimitive.attributes.end())
                    {
                        tinygltf::Accessor const& accessor =
                            input.accessors[glTFPrimitive.attributes.find("TEXCOORD_0")->second];
                        tinygltf::BufferView const& view = input.bufferViews[accessor.bufferView];
                        texCoordsBuffer                  = reinterpret_cast<float const*>(
                            &(input.buffers[view.buffer].data[accessor.byteOffset + view.byteOffset]));

                        material.flags |= std::to_underlying(MaterialFeatures::TexcoordVertexAttribute);
                    }

                    if (glTFPrimitive.attributes.find("TANGENT") != glTFPrimitive.attributes.end())
                    {
                        tinygltf::Accessor const& accessor =
                            input.accessors[glTFPrimitive.attributes.find("TANGENT")->second];
                        tinygltf::BufferView const& view = input.bufferViews[accessor.bufferView];
                        tangentsBuffer                   = reinterpret_cast<float const*>(
                            &(input.buffers[view.buffer].data[accessor.byteOffset + view.byteOffset]));

                        material.flags |= std::to_underlying(MaterialFeatures::TangentVertexAttribute);
                    }

                    // Append data to model's vertex buffer
                    for (size_t v = 0; v < vertexCount; v++)
                    {
                        Vertex vert {};
                        vert.position = glm::make_vec3(&positionBuffer[v * 3]);
                        vert.normal   = glm::normalize(glm::vec3(
                            normalsBuffer ? glm::make_vec3(&normalsBuffer[v * 3]) : glm::vec3(0.0f)));
                        vert.tangent = tangentsBuffer ? glm::make_vec4(&tangentsBuffer[v * 4]) : glm::vec4(0);

                        if (texCoordsBuffer)
                        {
                            glm::vec2 uv_vec = glm::make_vec2(&texCoordsBuffer[v * 2]);
                            vert.uv_x        = uv_vec.x;
                            vert.uv_y        = uv_vec.y;
                        }
                        vertexBuffer.push_back(vert);
                    }
                }

                // Indices
                {
                    tinygltf::Accessor const& accessor     = input.accessors[glTFPrimitive.indices];
                    tinygltf::BufferView const& bufferView = input.bufferViews[accessor.bufferView];
                    tinygltf::Buffer const& buffer         = input.buffers[bufferView.buffer];

                    indexCount += static_cast<uint32_t>(accessor.count);

                    // glTF supports different component types of indices
                    switch (accessor.componentType)
                    {
                        case TINYGLTF_PARAMETER_TYPE_UNSIGNED_INT:
                            {
                                uint32_t const* buf = reinterpret_cast<uint32_t const*>(
                                    &buffer.data[accessor.byteOffset + bufferView.byteOffset]);
                                for (size_t index = 0; index < accessor.count; index++)
                                {
                                    indexBuffer.push_back(buf[index] + vertexStart);
                                }
                                break;
                            }
                        case TINYGLTF_PARAMETER_TYPE_UNSIGNED_SHORT:
                            {
                                uint16_t const* buf = reinterpret_cast<uint16_t const*>(
                                    &buffer.data[accessor.byteOffset + bufferView.byteOffset]);
                                for (size_t index = 0; index < accessor.count; index++)
                                {
                                    indexBuffer.push_back(buf[index] + vertexStart);
                                }
                                break;
                            }
                        case TINYGLTF_PARAMETER_TYPE_UNSIGNED_BYTE:
                            {
                                uint8_t const* buf = reinterpret_cast<uint8_t const*>(
                                    &buffer.data[accessor.byteOffset + bufferView.byteOffset]);
                                for (size_t index = 0; index < accessor.count; index++)
                                {
                                    indexBuffer.push_back(buf[index] + vertexStart);
                                }
                                break;
                            }
                        default:
                            MC_ASSERT_MSG(false, "Unsupported index type");
                            return;
                    }
                }

                node->mesh.primitives.push_back({
                    .firstIndex    = firstIndex,
                    .indexCount    = indexCount,
                    .materialIndex = glTFPrimitive.material,
                });
            }
        }

        if (parent)
        {
            parent->children.push_back(node);
        }
        else
        {
            m_sceneResources.nodes.push_back(node);
        }
    }

    void RendererBackend::drawNode(vk::CommandBuffer commandBuffer,
                                   vk::PipelineLayout pipelineLayout,
                                   GltfNode* node)
    {
        if (node->mesh.primitives.size() > 0)
        {
            glm::mat4 nodeTransform = node->transformation;
            GltfNode* currentParent = node->parent;

            // TODO(aether) prolly precalculate this? profile it tho
            while (currentParent)
            {
                nodeTransform = currentParent->transformation * nodeTransform;
                currentParent = currentParent->parent;
            }

            for (Primitive& primitive : node->mesh.primitives)
            {
                if (primitive.indexCount == 0)
                {
                    return;
                }

                m_stats.drawcall_count++;
                m_stats.triangle_count += primitive.indexCount / 3;

                GPUDrawPushConstants pushConstants {
                    .model         = nodeTransform,
                    .materialIndex = static_cast<uint32_t>(primitive.materialIndex),
                };

                commandBuffer.pushConstants(pipelineLayout,
                                            vk::ShaderStageFlagBits::eVertex |
                                                vk::ShaderStageFlagBits::eFragment,
                                            0,
                                            sizeof(GPUDrawPushConstants),
                                            &pushConstants);

                commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, m_texturedPipeline);

                commandBuffer.bindDescriptorSets(
                    vk::PipelineBindPoint::eGraphics,
                    m_texturedPipelineLayout,
                    0,
                    { m_sceneDataDescriptors, m_sceneResources.materialDescriptors[primitive.materialIndex] },
                    {});

                {
#if PROFILED
                    auto& tracyCtx = m_frameResources[m_currentFrame].tracyContext;
#endif

                    TracyVkZone(tracyCtx, commandBuffer, "draw call");

                    commandBuffer.drawIndexed(primitive.indexCount, 1, primitive.firstIndex, 0, 0);
                }
            }
        }
        for (auto& child : node->children)
        {
            drawNode(commandBuffer, pipelineLayout, child);
        }
    };

    void RendererBackend::drawGltf(vk::CommandBuffer commandBuffer, vk::PipelineLayout pipelineLayout)
    {
        commandBuffer.bindIndexBuffer(m_sceneResources.indexBuffer, 0, vk::IndexType::eUint32);

        for (auto& node : m_sceneResources.nodes)
        {
            drawNode(commandBuffer, pipelineLayout, node);
        }
    }
}  // namespace renderer::backend
