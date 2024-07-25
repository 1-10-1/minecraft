#include "mc/renderer/backend/buffer.hpp"
#include <mc/renderer/backend/allocator.hpp>
#include <mc/renderer/backend/gltfloader.hpp>
#include <mc/renderer/backend/renderer_backend.hpp>
#include <mc/utils.hpp>

#include <cstring>
#include <fstream>

#include <basisu_transcoder.h>
#include <glm/gtc/type_ptr.hpp>
#include <vulkan/vulkan_enums.hpp>
#include <vulkan/vulkan_structs.hpp>

namespace renderer::backend
{
    // We use a custom image loading function with tinyglTF, so we can do custom stuff loading ktx textures
    bool loadImageDataFunc(tinygltf::Image* image,
                           int const imageIndex,
                           std::string* error,
                           std::string* warning,
                           int req_width,
                           int req_height,
                           unsigned char const* bytes,
                           int size,
                           void* userData)
    {
        // KTX files will be handled by our own code
        if (image->uri.find_last_of(".") != std::string::npos)
        {
            if (image->uri.substr(image->uri.find_last_of(".") + 1) == "ktx2")
            {
                return true;
            }
        }

        return tinygltf::LoadImageData(
            image, imageIndex, error, warning, req_width, req_height, bytes, size, userData);
    }

    // Bounding box

    BoundingBox BoundingBox::getAABB(glm::mat4 m)
    {
        glm::vec3 min = glm::vec3(m[3]);
        glm::vec3 max = min;
        glm::vec3 v0, v1;

        glm::vec3 right = glm::vec3(m[0]);
        v0              = right * this->min.x;
        v1              = right * this->max.x;
        min += glm::min(v0, v1);
        max += glm::max(v0, v1);

        glm::vec3 up = glm::vec3(m[1]);
        v0           = up * this->min.y;
        v1           = up * this->max.y;
        min += glm::min(v0, v1);
        max += glm::max(v0, v1);

        glm::vec3 back = glm::vec3(m[2]);
        v0             = back * this->min.z;
        v1             = back * this->max.z;
        min += glm::min(v0, v1);
        max += glm::max(v0, v1);

        return BoundingBox(min, max);
    }

    // Loads the image for this texture. Supports both glTF's web formats (jpg, png, embedded and external files) as well as external KTX2 files with basis universal texture compression
    GlTFTexture::GlTFTexture(Device& device,
                             Allocator& allocator,
                             CommandManager& cmdManager,
                             tinygltf::Image& gltfimage,
                             std::filesystem::path path,
                             TextureSampler textureSampler)
        : m_device { &device }, m_allocator { &allocator }, m_commandManager { &cmdManager }
    {
        // KTX2 files need to be handled explicitly
        bool isKtx2 = false;

        if (gltfimage.uri.find_last_of(".") != std::string::npos)
        {
            if (gltfimage.uri.substr(gltfimage.uri.find_last_of(".") + 1) == "ktx2")
            {
                isKtx2 = true;
            }
        }

        vk::Format format = vk::Format::eR8G8B8A8Unorm;

        uint32_t width, height, mipLevels;

        if (isKtx2)
        {
            // Image is KTX2 using basis universal compression. Those images need to be loaded from disk and will be transcoded to a native GPU format

            basist::ktx2_transcoder ktxTranscoder;

            std::filesystem::path const filename = path / gltfimage.uri;

            std::ifstream ifs(filename, std::ios::binary | std::ios::in | std::ios::ate);

            MC_ASSERT_MSG(ifs.is_open(), "Could not load the requested image file {}", filename.string());

            uint32_t inputDataSize = static_cast<uint32_t>(ifs.tellg());
            char* inputData        = new char[inputDataSize];

            ifs.seekg(0, std::ios::beg);
            ifs.read(inputData, inputDataSize);

            MC_ASSERT_MSG(ktxTranscoder.init(inputData, inputDataSize),
                          "Could not initialize ktx2 transcoder for image file {}",
                          filename.string());

            // Select target format based on device features (use uncompressed if none supported)
            auto targetFormat = basist::transcoder_texture_format::cTFRGBA32;

            auto deviceFeatures = device.getDeviceFeatures();

            auto formatSupported = [&device](vk::Format format)
            {
                vk::FormatProperties formatProperties = device.getFormatProperties(format);

                return ((formatProperties.optimalTilingFeatures & vk::FormatFeatureFlagBits::eTransferDst) &&
                        (formatProperties.optimalTilingFeatures & vk::FormatFeatureFlagBits::eSampledImage));
            };

            if (deviceFeatures.textureCompressionBC)
            {
                // BC7 is the preferred block compression if available
                if (auto desiredFormat = vk::Format::eBc7UnormBlock; formatSupported(desiredFormat))
                {
                    targetFormat = basist::transcoder_texture_format::cTFBC7_RGBA;
                    format       = desiredFormat;
                }
                else if (auto desiredFormat = vk::Format::eBc3SrgbBlock; formatSupported(desiredFormat))
                {
                    targetFormat = basist::transcoder_texture_format::cTFBC3_RGBA;
                    format       = desiredFormat;
                }
            }

            // Adaptive scalable texture compression
            if (deviceFeatures.textureCompressionASTC_LDR)
            {
                if (auto desiredFormat = vk::Format::eAstc4x4SrgbBlock; formatSupported(desiredFormat))
                {
                    targetFormat = basist::transcoder_texture_format::cTFASTC_4x4_RGBA;
                    format       = desiredFormat;
                }
            }

            // Ericsson texture compression
            if (deviceFeatures.textureCompressionETC2)
            {
                if (auto desiredFormat = vk::Format::eEtc2R8G8B8SrgbBlock; formatSupported(desiredFormat))
                {
                    targetFormat = basist::transcoder_texture_format::cTFETC2_RGBA;
                    format       = desiredFormat;
                }
            }

            // TODO(aether) PowerVR texture compression support needs to be checked
            // via an extension (VK_IMG_FORMAT_PVRTC_EXTENSION_NAME)

            bool const targetFormatIsUncompressed =
                basist::basis_transcoder_format_is_uncompressed(targetFormat);

            std::vector<basist::ktx2_image_level_info> levelInfos(ktxTranscoder.get_levels());

            mipLevels = ktxTranscoder.get_levels();

            // Query image level information that we need later on for several calculations
            // We only support 2D images (no cube maps or layered images)
            for (uint32_t i = 0; i < mipLevels; i++)
            {
                ktxTranscoder.get_image_level_info(levelInfos[i], i, 0, 0);
            }

            width  = levelInfos[0].m_orig_width;
            height = levelInfos[0].m_orig_height;

            // Create one staging buffer large enough to hold all uncompressed image levels
            uint32_t const bytesPerBlockOrPixel = basist::basis_get_bytes_per_block_or_pixel(targetFormat);
            uint32_t numBlocksOrPixels          = 0;
            VkDeviceSize totalBufferSize        = 0;
            for (uint32_t i = 0; i < mipLevels; i++)
            {
                // Size calculations differ for compressed/uncompressed formats
                numBlocksOrPixels = targetFormatIsUncompressed
                                        ? levelInfos[i].m_orig_width * levelInfos[i].m_orig_height
                                        : levelInfos[i].m_total_blocks;
                totalBufferSize += numBlocksOrPixels * bytesPerBlockOrPixel;
            }

            GPUBuffer stagingBuffer(*m_device,
                                    *m_allocator,
                                    "Image staging buffer (compressed)",
                                    totalBufferSize,
                                    vk::BufferUsageFlagBits::eTransferSrc,
                                    VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                                    VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                                        VMA_ALLOCATION_CREATE_MAPPED_BIT);

            unsigned char* buffer    = new unsigned char[totalBufferSize];
            unsigned char* bufferPtr = &buffer[0];

            MC_ASSERT_MSG(ktxTranscoder.start_transcoding(),
                          "Could not start transcoding for image file {}",
                          filename.string());

            // Transcode all mip levels into the staging buffer
            for (uint32_t i = 0; i < mipLevels; i++)
            {
                // Size calculations differ for compressed/uncompressed formats
                numBlocksOrPixels = targetFormatIsUncompressed
                                        ? levelInfos[i].m_orig_width * levelInfos[i].m_orig_height
                                        : levelInfos[i].m_total_blocks;

                MC_ASSERT_MSG(ktxTranscoder.transcode_image_level(
                                  i, 0, 0, bufferPtr, numBlocksOrPixels, targetFormat, 0),
                              "Could not transcode the requested image file {}",
                              filename.string());

                bufferPtr += numBlocksOrPixels * bytesPerBlockOrPixel;
            }

            std::memcpy(stagingBuffer.getMappedData(), buffer, totalBufferSize);

            image = BasicImage(device,
                               allocator,
                               { width, height },
                               format,
                               vk::SampleCountFlagBits::e1,
                               vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst |
                                   vk::ImageUsageFlagBits::eSampled,
                               vk::ImageAspectFlagBits::eColor,
                               mipLevels,
                               std::format("Compressed gltf texture ({})", gltfimage.uri));

            ScopedCommandBuffer copyCmd(
                device, cmdManager.getTransferCmdPool(), device.getTransferQueue(), true);

            vk::ImageSubresourceRange subresourceRange = {
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .levelCount = mipLevels,
                .layerCount = 1,
            };

            vk::ImageMemoryBarrier imageMemoryBarrier {
                .srcAccessMask    = vk::AccessFlagBits::eNone,
                .dstAccessMask    = vk::AccessFlagBits::eTransferWrite,
                .oldLayout        = vk::ImageLayout::eUndefined,
                .newLayout        = vk::ImageLayout::eTransferDstOptimal,
                .image            = image,
                .subresourceRange = subresourceRange,
            };

            copyCmd->pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
                                     vk::PipelineStageFlagBits::eAllCommands,
                                     {},
                                     {},
                                     {},
                                     { imageMemoryBarrier });

            // Transcode and copy all image levels
            vk::DeviceSize bufferOffset = 0;

            for (uint32_t i = 0; i < mipLevels; i++)
            {
                // Size calculations differ for compressed/uncompressed formats
                numBlocksOrPixels   = targetFormatIsUncompressed
                                          ? levelInfos[i].m_orig_width * levelInfos[i].m_orig_height
                                          : levelInfos[i].m_total_blocks;
                uint32_t outputSize = numBlocksOrPixels * bytesPerBlockOrPixel;

                vk::BufferImageCopy bufferCopyRegion {
                    .bufferOffset = bufferOffset,
                    .imageSubresource {
                                       .aspectMask     = vk::ImageAspectFlagBits::eColor,
                                       .mipLevel       = i,
                                       .baseArrayLayer = 0,
                                       .layerCount     = 1,
                                       },
                    .imageExtent {
                                       .width  = levelInfos[i].m_orig_width,
                                       .height = levelInfos[i].m_orig_height,
                                       .depth  = 1,
                                       },
                };

                copyCmd->copyBufferToImage(
                    stagingBuffer, image, vk::ImageLayout::eTransferDstOptimal, { bufferCopyRegion });

                bufferOffset += outputSize;
            }

            imageMemoryBarrier.oldLayout        = vk::ImageLayout::eTransferDstOptimal;
            imageMemoryBarrier.newLayout        = vk::ImageLayout::eShaderReadOnlyOptimal;
            imageMemoryBarrier.srcAccessMask    = vk::AccessFlagBits::eTransferWrite;
            imageMemoryBarrier.dstAccessMask    = vk::AccessFlagBits::eShaderRead;
            imageMemoryBarrier.image            = image;
            imageMemoryBarrier.subresourceRange = subresourceRange;

            copyCmd->pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
                                     vk::PipelineStageFlagBits::eAllCommands,
                                     {},
                                     {},
                                     {},
                                     { imageMemoryBarrier });

            copyCmd.flush();

            delete[] buffer;
            delete[] inputData;
        }
        else
        {
            // Image is a basic glTF format like png or jpg and can be loaded directly via tinyglTF
            unsigned char* buffer     = nullptr;
            vk::DeviceSize bufferSize = 0;
            bool deleteBuffer         = false;

            if (gltfimage.component == 3)
            {
                // Most devices don't support RGB only on Vulkan so convert if necessary
                bufferSize          = gltfimage.width * gltfimage.height * 4;
                buffer              = new unsigned char[bufferSize];
                unsigned char* rgba = buffer;
                unsigned char* rgb  = &gltfimage.image[0];
                for (int32_t i = 0; i < gltfimage.width * gltfimage.height; ++i)
                {
                    for (int32_t j = 0; j < 3; ++j)
                    {
                        rgba[j] = rgb[j];
                    }
                    rgba += 4;
                    rgb += 3;
                }
                deleteBuffer = true;
            }
            else
            {
                buffer     = &gltfimage.image[0];
                bufferSize = gltfimage.image.size();
            }

            width     = gltfimage.width;
            height    = gltfimage.height;
            mipLevels = static_cast<uint32_t>(floor(log2(std::max(width, height))) + 1.0);

            MC_ASSERT_MSG(device.getFormatProperties(format).optimalTilingFeatures &
                              (vk::FormatFeatureFlagBits::eBlitSrc | vk::FormatFeatureFlagBits::eBlitDst),
                          "Blitting is not supported");

            GPUBuffer stagingBuffer(*m_device,
                                    *m_allocator,
                                    "Image staging buffer (uncompressed)",
                                    bufferSize,
                                    vk::BufferUsageFlagBits::eTransferSrc,
                                    VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                                    VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                                        // TODO(aether) maybe unmap it asap if thats gonna be benefitial
                                        VMA_ALLOCATION_CREATE_MAPPED_BIT);

            uint8_t* data = reinterpret_cast<uint8_t*>(stagingBuffer.getMappedData());
            std::memcpy(data, buffer, bufferSize);

            if (deleteBuffer)
            {
                delete[] buffer;
            }

            image = BasicImage(device,
                               allocator,
                               { width, height },
                               format,
                               vk::SampleCountFlagBits::e1,
                               vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst |
                                   vk::ImageUsageFlagBits::eSampled,
                               vk::ImageAspectFlagBits::eColor,
                               mipLevels,
                               std::format("Uncompressed gltf texture ({})", gltfimage.uri));

            ScopedCommandBuffer cmdBuf(
                device, cmdManager.getTransferCmdPool(), device.getTransferQueue(), true);

            vk::ImageSubresourceRange subresourceRange = {
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .levelCount = 1,
                .layerCount = 1,
            };

            {
                vk::ImageMemoryBarrier imageMemoryBarrier {
                    .srcAccessMask    = vk::AccessFlagBits::eNone,
                    .dstAccessMask    = vk::AccessFlagBits::eTransferWrite,
                    .oldLayout        = vk::ImageLayout::eUndefined,
                    .newLayout        = vk::ImageLayout::eTransferDstOptimal,
                    .image            = image,
                    .subresourceRange = subresourceRange,
                };

                cmdBuf->pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
                                        vk::PipelineStageFlagBits::eAllCommands,
                                        {},
                                        {},
                                        {},
                                        { imageMemoryBarrier });
            }

            // clang-format off
            vk::BufferImageCopy bufferCopyRegion = {
                .imageSubresource {
                    .aspectMask     = vk::ImageAspectFlagBits::eColor,
                    .mipLevel       = 0,
                    .baseArrayLayer = 0,
                    .layerCount     = 1,
                },
                .imageExtent {
                    .width = width,
                    .height = height,
                    .depth = 1
                },
            };
            // clang-format on

            cmdBuf->copyBufferToImage(
                stagingBuffer, image, vk::ImageLayout::eTransferDstOptimal, { bufferCopyRegion });

            cmdBuf->pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
                                    vk::PipelineStageFlagBits::eAllCommands,
                                    {
            },
                                    {},
                                    {},
                                    { vk::ImageMemoryBarrier {
                                        .srcAccessMask    = vk::AccessFlagBits::eTransferWrite,
                                        .dstAccessMask    = vk::AccessFlagBits::eTransferRead,
                                        .oldLayout        = vk::ImageLayout::eTransferDstOptimal,
                                        .newLayout        = vk::ImageLayout::eTransferSrcOptimal,
                                        .image            = image,
                                        .subresourceRange = subresourceRange,
                                    } });

            // Generate the mip chain (glTF uses jpg and png, so we need to create this manually)
            for (uint32_t i = 1; i < mipLevels; i++)
            {
                // clang-format off
                vk::ImageBlit imageBlit
                {
                    .srcSubresource {
                        .aspectMask = vk::ImageAspectFlagBits::eColor,
                        .mipLevel   = i - 1,
                        .layerCount = 1,
                    },

                    .srcOffsets = std::array {
                        vk::Offset3D(0, 0, 0),
                        vk::Offset3D()
                            .setX(width >> (i - 1))
                            .setY(height >> (i - 1))
                            .setZ(1)
                    },

                    .dstSubresource {
                        .aspectMask = vk::ImageAspectFlagBits::eColor,
                        .mipLevel   = i,
                        .layerCount = 1,
                    },

                    .dstOffsets = std::array {
                        vk::Offset3D(0, 0, 0),
                        vk::Offset3D()
                            .setX(width >> i)
                            .setY(height >> i)
                            .setZ(1)
                    }
                    // clang-format on
                };

                vk::ImageSubresourceRange mipSubRange {
                    .aspectMask   = vk::ImageAspectFlagBits::eColor,
                    .baseMipLevel = i,
                    .levelCount   = 1,
                    .layerCount   = 1,
                };

                cmdBuf->pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                                        vk::PipelineStageFlagBits::eTransfer,
                                        {
                },
                                        {},
                                        {},
                                        { vk::ImageMemoryBarrier {
                                            .srcAccessMask    = vk::AccessFlagBits::eNone,
                                            .dstAccessMask    = vk::AccessFlagBits::eTransferWrite,
                                            .oldLayout        = vk::ImageLayout::eUndefined,
                                            .newLayout        = vk::ImageLayout::eTransferDstOptimal,
                                            .image            = image,
                                            .subresourceRange = mipSubRange,
                                        } });

                cmdBuf->blitImage(image,
                                  vk::ImageLayout::eTransferSrcOptimal,
                                  image,
                                  vk::ImageLayout::eTransferDstOptimal,
                                  { imageBlit },
                                  vk::Filter::eLinear);

                cmdBuf->pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                                        vk::PipelineStageFlagBits::eTransfer,
                                        {
                },
                                        {},
                                        {},
                                        { vk::ImageMemoryBarrier {
                                            .srcAccessMask    = vk::AccessFlagBits::eTransferWrite,
                                            .dstAccessMask    = vk::AccessFlagBits::eTransferRead,
                                            .oldLayout        = vk::ImageLayout::eTransferDstOptimal,
                                            .newLayout        = vk::ImageLayout::eTransferSrcOptimal,
                                            .image            = image,
                                            .subresourceRange = mipSubRange,
                                        } });
            }

            subresourceRange.levelCount = mipLevels;

            cmdBuf->pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
                                    vk::PipelineStageFlagBits::eAllCommands,
                                    {
            },
                                    {},
                                    {},
                                    { vk::ImageMemoryBarrier {
                                        .srcAccessMask    = vk::AccessFlagBits::eTransferWrite,
                                        .dstAccessMask    = vk::AccessFlagBits::eShaderRead,
                                        .oldLayout        = vk::ImageLayout::eTransferSrcOptimal,
                                        .newLayout        = vk::ImageLayout::eShaderReadOnlyOptimal,
                                        .image            = image,
                                        .subresourceRange = subresourceRange,
                                    } });
        }

        sampler = device->createSampler(vk::SamplerCreateInfo {
                      .magFilter        = textureSampler.magFilter,
                      .minFilter        = textureSampler.minFilter,
                      .mipmapMode       = vk::SamplerMipmapMode::eLinear,
                      .addressModeU     = textureSampler.addressModeU,
                      .addressModeV     = textureSampler.addressModeV,
                      .addressModeW     = textureSampler.addressModeW,
                      .anisotropyEnable = VK_TRUE,
                      .maxAnisotropy    = 8.0f,
                      .compareOp        = vk::CompareOp::eNever,
                      .maxLod           = static_cast<float>(mipLevels),
                      .borderColor      = vk::BorderColor::eFloatOpaqueWhite,
                  }) >>
                  ResultChecker();
    }

    // Primitive
    Primitive::Primitive(uint32_t firstIndex,
                         uint32_t indexCount,
                         uint32_t vertexCount,
                         uint32_t materialIndex)
        : firstIndex { firstIndex },
          indexCount { indexCount },
          vertexCount { vertexCount },
          materialIndex { materialIndex }
    {
        hasIndices = indexCount > 0;
    };

    void Primitive::setBoundingBox(glm::vec3 min, glm::vec3 max)
    {
        bb.min   = min;
        bb.max   = max;
        bb.valid = true;
    }

    // Mesh
    Mesh::Mesh(Allocator& allocator, glm::mat4 matrix)
    {
        uniformBlock.matrix = matrix;

        // TODO(aether) maybe use REBAR?
        // https://gpuopen-librariesandsdks.github.io/VulkanMemoryAllocator/html/usage_patterns.html#usage_patterns_advanced_data_uploading
        // We'll need to create a separate copying mechanism in GPUBuffer where we check if the memory
        // resides in a device local + host visible memory, and if not, make sure we flush any writes and
        // also copy the staging buffer to the device local one.
        uniformBuffer.buffer = GPUBuffer(allocator,
                                         sizeof(uniformBlock),
                                         vk::BufferUsageFlagBits::eUniformBuffer,
                                         VMA_MEMORY_USAGE_AUTO,
                                         VMA_ALLOCATION_CREATE_MAPPED_BIT |
                                             VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);

        uniformBuffer.mapped = uniformBuffer.buffer.getMappedData();

        std::memcpy(uniformBuffer.mapped, &uniformBlock, sizeof(UniformBlock));

        uniformBuffer.descriptor = { uniformBuffer.buffer.get(), 0, sizeof(uniformBlock) };
    };

    void Mesh::setBoundingBox(glm::vec3 min, glm::vec3 max)
    {
        bb.min   = min;
        bb.max   = max;
        bb.valid = true;
    }

    // Node
    glm::mat4 Node::localMatrix()
    {
        if (!useCachedMatrix)
        {
            cachedLocalMatrix = glm::translate(glm::mat4(1.0f), translation) *
                                glm::mat4(glm::quat { static_cast<float>(rotation.w),
                                                      static_cast<float>(rotation.x),
                                                      static_cast<float>(rotation.y),
                                                      static_cast<float>(rotation.z) }) *
                                glm::scale(glm::mat4(1.0f), scale) * matrix;
        };

        return cachedLocalMatrix;
    }

    glm::mat4 Node::getMatrix()
    {
        // Use a simple caching algorithm to avoid having to recalculate matrices to often while traversing the node hierarchy
        if (!useCachedMatrix)
        {
            glm::mat4 m = localMatrix();
            Node* p     = parent;

            while (p)
            {
                m = p->localMatrix() * m;
                p = p->parent;
            }

            cachedMatrix    = m;
            useCachedMatrix = true;

            return m;
        }
        else
        {
            return cachedMatrix;
        }
    }

    void Node::update()
    {
        useCachedMatrix = false;

        if (mesh)
        {
            glm::mat4 m               = getMatrix();
            mesh->uniformBlock.matrix = m;

            if (skin)
            {
                // Update joint matrices
                glm::mat4 inverseTransform = glm::inverse(m);
                size_t numJoints           = std::min(utils::size(skin->joints), kMaxNumJoints);

                for (size_t i = 0; i < numJoints; i++)
                {
                    Node* jointNode = skin->joints[i];
                    glm::mat4 jointMat =
                        inverseTransform * jointNode->getMatrix() * skin->inverseBindMatrices[i];
                    mesh->uniformBlock.jointMatrix[i] = jointMat;
                }

                mesh->uniformBlock.jointcount = static_cast<uint32_t>(numJoints);
                std::memcpy(mesh->uniformBuffer.mapped, &mesh->uniformBlock, sizeof(mesh->uniformBlock));
            }
            else
            {
                std::memcpy(mesh->uniformBuffer.mapped, &m, sizeof(glm::mat4));
            }
        }

        for (auto& child : children)
        {
            child->update();
        }
    }

    // AnimationSampler

    // Cube spline interpolation function used for translate/scale/rotate with cubic spline animation samples
    // Details on how this works can be found in the specs https://github.com/KhronosGroup/glTF/tree/master/specification/2.0#appendix-c-spline-interpolation
    glm::vec4 AnimationSampler::cubicSplineInterpolation(size_t index, float time, uint32_t stride)
    {
        float delta          = inputs[index + 1] - inputs[index];
        float t              = (time - inputs[index]) / delta;
        size_t const current = index * stride * 3;
        size_t const next    = (index + 1) * stride * 3;
        size_t const A       = 0;
        size_t const V       = stride * 1;
        // [[maybe_unused]] size_t const B       = stride * 2;

        float t2 = glm::pow(t, 2);
        float t3 = glm::pow(t, 3);

        glm::vec4 pt { 0.0f };
        for (uint32_t i = 0; i < stride; i++)
        {
            float p0 = outputs[current + i + V];          // starting point at t = 0
            float m0 = delta * outputs[current + i + A];  // scaled starting tangent at t = 0
            float p1 = outputs[next + i + V];             // ending point at t = 1
            // [[maybe_unused]] float m1 = delta * outputs[next + i + B];     // scaled ending tangent at t = 1
            pt[i] = ((2.f * t3 - 3.f * t2 + 1.f) * p0) + ((t3 - 2.f * t2 + t) * m0) +
                    ((-2.f * t3 + 3.f * t2) * p1) + ((t3 - t2) * m0);
        }
        return pt;
    }

    // Calculates the translation of this sampler for the given node at a given time point depending on the interpolation type
    void AnimationSampler::translate(size_t index, float time, Node* node)
    {
        switch (interpolation)
        {
            case AnimationSampler::InterpolationType::LINEAR:
                {
                    float u = std::max(0.0f, time - inputs[index]) / (inputs[index + 1] - inputs[index]);
                    node->translation =
                        glm::make_vec3(glm::mix(outputsVec4[index], outputsVec4[index + 1], u));
                    break;
                }
            case AnimationSampler::InterpolationType::STEP:
                {
                    node->translation = glm::make_vec3(outputsVec4[index]);
                    break;
                }
            case AnimationSampler::InterpolationType::CUBICSPLINE:
                {
                    node->translation = glm::make_vec3(cubicSplineInterpolation(index, time, 3));
                    break;
                }
        }
    }

    // Calculates the scale of this sampler for the given node at a given time point depending on the interpolation type
    void AnimationSampler::scale(size_t index, float time, Node* node)
    {
        switch (interpolation)
        {
            case AnimationSampler::InterpolationType::LINEAR:
                {
                    float u     = std::max(0.0f, time - inputs[index]) / (inputs[index + 1] - inputs[index]);
                    node->scale = glm::make_vec3(glm::mix(outputsVec4[index], outputsVec4[index + 1], u));
                    break;
                }
            case AnimationSampler::InterpolationType::STEP:
                {
                    node->scale = glm::make_vec3(outputsVec4[index]);
                    break;
                }
            case AnimationSampler::InterpolationType::CUBICSPLINE:
                {
                    node->scale = glm::make_vec3(cubicSplineInterpolation(index, time, 3));
                    break;
                }
        }
    }

    // Calculates the rotation of this sampler for the given node at a given time point depending on the interpolation type
    void AnimationSampler::rotate(size_t index, float time, Node* node)
    {
        switch (interpolation)
        {
            case AnimationSampler::InterpolationType::LINEAR:
                {
                    float u = std::max(0.0f, time - inputs[index]) / (inputs[index + 1] - inputs[index]);
                    glm::quat q1;
                    q1.x = outputsVec4[index].x;
                    q1.y = outputsVec4[index].y;
                    q1.z = outputsVec4[index].z;
                    q1.w = outputsVec4[index].w;
                    glm::quat q2;
                    q2.x           = outputsVec4[index + 1].x;
                    q2.y           = outputsVec4[index + 1].y;
                    q2.z           = outputsVec4[index + 1].z;
                    q2.w           = outputsVec4[index + 1].w;
                    node->rotation = glm::normalize(glm::slerp(q1, q2, u));
                    break;
                }
            case AnimationSampler::InterpolationType::STEP:
                {
                    glm::quat q1;
                    q1.x           = outputsVec4[index].x;
                    q1.y           = outputsVec4[index].y;
                    q1.z           = outputsVec4[index].z;
                    q1.w           = outputsVec4[index].w;
                    node->rotation = q1;
                    break;
                }
            case AnimationSampler::InterpolationType::CUBICSPLINE:
                {
                    glm::vec4 rot = cubicSplineInterpolation(index, time, 4);
                    glm::quat q;
                    q.x            = rot.x;
                    q.y            = rot.y;
                    q.z            = rot.z;
                    q.w            = rot.w;
                    node->rotation = glm::normalize(q);
                    break;
                }
        }
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

    void Model::loadNode(Node* parent,
                         tinygltf::Node const& node,
                         uint32_t nodeIndex,
                         tinygltf::Model const& model,
                         LoaderInfo& loaderInfo,
                         float globalscale)
    {
        Node* newNode      = new Node {};
        newNode->index     = nodeIndex;
        newNode->parent    = parent;
        newNode->name      = node.name;
        newNode->skinIndex = node.skin;
        newNode->matrix    = glm::mat4(1.0f);

        // Generate local node matrix
        glm::vec3 translation = glm::vec3(0.0f);
        if (node.translation.size() == 3)
        {
            translation          = glm::make_vec3(node.translation.data());
            newNode->translation = translation;
        }
        if (node.rotation.size() == 4)
        {
            newNode->rotation = glm::make_quat(node.rotation.data());
        }
        glm::vec3 scale = glm::vec3(1.0f);
        if (node.scale.size() == 3)
        {
            scale          = glm::make_vec3(node.scale.data());
            newNode->scale = scale;
        }
        if (node.matrix.size() == 16)
        {
            newNode->matrix = glm::make_mat4x4(node.matrix.data());
        }

        // Node with children
        if (node.children.size() > 0)
        {
            for (size_t i = 0; i < node.children.size(); i++)
            {
                loadNode(
                    newNode, model.nodes[node.children[i]], node.children[i], model, loaderInfo, globalscale);
            }
        }

        // Node contains mesh data
        if (node.mesh > -1)
        {
            tinygltf::Mesh const mesh     = model.meshes[node.mesh];
            std::unique_ptr<Mesh> newMesh = std::make_unique<Mesh>(*allocator, newNode->matrix);

            for (size_t j = 0; j < mesh.primitives.size(); j++)
            {
                tinygltf::Primitive const& primitive = mesh.primitives[j];
                uint32_t vertexStart                 = static_cast<uint32_t>(loaderInfo.vertexPos);
                uint32_t indexStart                  = static_cast<uint32_t>(loaderInfo.indexPos);
                uint32_t indexCount                  = 0;
                uint32_t vertexCount                 = 0;
                glm::vec3 posMin {};
                glm::vec3 posMax {};
                bool hasSkin    = false;
                bool hasIndices = primitive.indices > -1;

                // Vertices
                {
                    float const* bufferPos          = nullptr;
                    float const* bufferTangents     = nullptr;
                    float const* bufferNormals      = nullptr;
                    float const* bufferTexCoordSet0 = nullptr;
                    float const* bufferTexCoordSet1 = nullptr;
                    float const* bufferColorSet0    = nullptr;
                    void const* bufferJoints        = nullptr;
                    float const* bufferWeights      = nullptr;

                    int posByteStride;
                    int tangentByteStride;
                    int normByteStride;
                    int uv0ByteStride;
                    int uv1ByteStride;
                    int color0ByteStride;
                    int jointByteStride;
                    int weightByteStride;

                    int jointComponentType;

                    // Position attribute is required
                    MC_ASSERT(primitive.attributes.find("POSITION") != primitive.attributes.end());

                    tinygltf::Accessor const& posAccessor =
                        model.accessors[primitive.attributes.find("POSITION")->second];

                    tinygltf::BufferView const& posView = model.bufferViews[posAccessor.bufferView];

                    bufferPos = reinterpret_cast<float const*>(
                        &(model.buffers[posView.buffer].data[posAccessor.byteOffset + posView.byteOffset]));

                    posMin = glm::vec3(
                        posAccessor.minValues[0], posAccessor.minValues[1], posAccessor.minValues[2]);

                    posMax = glm::vec3(
                        posAccessor.maxValues[0], posAccessor.maxValues[1], posAccessor.maxValues[2]);

                    vertexCount = static_cast<uint32_t>(posAccessor.count);

                    posByteStride = posAccessor.ByteStride(posView)
                                        ? (posAccessor.ByteStride(posView) / sizeof(float))
                                        : tinygltf::GetNumComponentsInType(TINYGLTF_TYPE_VEC3);

                    if (primitive.attributes.find("NORMAL") != primitive.attributes.end())
                    {
                        tinygltf::Accessor const& normAccessor =
                            model.accessors[primitive.attributes.find("NORMAL")->second];
                        tinygltf::BufferView const& normView = model.bufferViews[normAccessor.bufferView];
                        bufferNormals                        = reinterpret_cast<float const*>(
                            &(model.buffers[normView.buffer]
                                  .data[normAccessor.byteOffset + normView.byteOffset]));
                        normByteStride = normAccessor.ByteStride(normView)
                                             ? (normAccessor.ByteStride(normView) / sizeof(float))
                                             : tinygltf::GetNumComponentsInType(TINYGLTF_TYPE_VEC3);
                    }

                    if (primitive.attributes.find("TANGENT") != primitive.attributes.end())
                    {
                        tinygltf::Accessor const& tanAccessor =
                            model.accessors[primitive.attributes.find("TANGENT")->second];

                        tinygltf::BufferView const& tanView = model.bufferViews[tanAccessor.bufferView];

                        bufferTangents = reinterpret_cast<float const*>(&(
                            model.buffers[tanView.buffer].data[tanAccessor.byteOffset + tanView.byteOffset]));

                        tangentByteStride = tanAccessor.ByteStride(tanView)
                                                ? (tanAccessor.ByteStride(tanView) / sizeof(float))
                                                : tinygltf::GetNumComponentsInType(TINYGLTF_TYPE_VEC3);
                    }

                    // UVs
                    if (primitive.attributes.find("TEXCOORD_0") != primitive.attributes.end())
                    {
                        tinygltf::Accessor const& uvAccessor =
                            model.accessors[primitive.attributes.find("TEXCOORD_0")->second];

                        tinygltf::BufferView const& uvView = model.bufferViews[uvAccessor.bufferView];

                        bufferTexCoordSet0 = reinterpret_cast<float const*>(
                            &(model.buffers[uvView.buffer].data[uvAccessor.byteOffset + uvView.byteOffset]));

                        uv0ByteStride = uvAccessor.ByteStride(uvView)
                                            ? (uvAccessor.ByteStride(uvView) / sizeof(float))
                                            : tinygltf::GetNumComponentsInType(TINYGLTF_TYPE_VEC2);
                    }

                    if (primitive.attributes.find("TEXCOORD_1") != primitive.attributes.end())
                    {
                        tinygltf::Accessor const& uvAccessor =
                            model.accessors[primitive.attributes.find("TEXCOORD_1")->second];

                        tinygltf::BufferView const& uvView = model.bufferViews[uvAccessor.bufferView];

                        bufferTexCoordSet1 = reinterpret_cast<float const*>(
                            &(model.buffers[uvView.buffer].data[uvAccessor.byteOffset + uvView.byteOffset]));

                        uv1ByteStride = uvAccessor.ByteStride(uvView)
                                            ? (uvAccessor.ByteStride(uvView) / sizeof(float))
                                            : tinygltf::GetNumComponentsInType(TINYGLTF_TYPE_VEC2);
                    }

                    // Vertex colors
                    if (primitive.attributes.find("COLOR_0") != primitive.attributes.end())
                    {
                        tinygltf::Accessor const& accessor =
                            model.accessors[primitive.attributes.find("COLOR_0")->second];

                        tinygltf::BufferView const& view = model.bufferViews[accessor.bufferView];

                        bufferColorSet0 = reinterpret_cast<float const*>(
                            &(model.buffers[view.buffer].data[accessor.byteOffset + view.byteOffset]));

                        color0ByteStride = accessor.ByteStride(view)
                                               ? (accessor.ByteStride(view) / sizeof(float))
                                               : tinygltf::GetNumComponentsInType(TINYGLTF_TYPE_VEC3);
                    }

                    // Skinning
                    // Joints
                    if (primitive.attributes.find("JOINTS_0") != primitive.attributes.end())
                    {
                        tinygltf::Accessor const& jointAccessor =
                            model.accessors[primitive.attributes.find("JOINTS_0")->second];

                        tinygltf::BufferView const& jointView = model.bufferViews[jointAccessor.bufferView];

                        bufferJoints = &(model.buffers[jointView.buffer]
                                             .data[jointAccessor.byteOffset + jointView.byteOffset]);

                        jointComponentType = jointAccessor.componentType;

                        jointByteStride = jointAccessor.ByteStride(jointView)
                                              ? (jointAccessor.ByteStride(jointView) /
                                                 tinygltf::GetComponentSizeInBytes(jointComponentType))
                                              : tinygltf::GetNumComponentsInType(TINYGLTF_TYPE_VEC4);
                    }

                    if (primitive.attributes.find("WEIGHTS_0") != primitive.attributes.end())
                    {
                        tinygltf::Accessor const& weightAccessor =
                            model.accessors[primitive.attributes.find("WEIGHTS_0")->second];

                        tinygltf::BufferView const& weightView = model.bufferViews[weightAccessor.bufferView];

                        bufferWeights = reinterpret_cast<float const*>(
                            &(model.buffers[weightView.buffer]
                                  .data[weightAccessor.byteOffset + weightView.byteOffset]));

                        weightByteStride = weightAccessor.ByteStride(weightView)
                                               ? (weightAccessor.ByteStride(weightView) / sizeof(float))
                                               : tinygltf::GetNumComponentsInType(TINYGLTF_TYPE_VEC4);
                    }

                    hasSkin = (bufferJoints && bufferWeights);

                    for (size_t v = 0; v < posAccessor.count; v++)
                    {
                        Vertex& vert = loaderInfo.vertexBuffer[loaderInfo.vertexPos] = Vertex {
                            .pos = glm::vec4(glm::make_vec3(&bufferPos[v * posByteStride]), 1.0f),

                            .normal = glm::normalize(
                                glm::vec3(bufferNormals ? glm::make_vec3(&bufferNormals[v * normByteStride])
                                                        : glm::vec3(0.0f))),

                            .uv0 = bufferTexCoordSet0 ? glm::make_vec2(&bufferTexCoordSet0[v * uv0ByteStride])
                                                      : glm::vec3(0.0f),

                            .uv1 = bufferTexCoordSet1 ? glm::make_vec2(&bufferTexCoordSet1[v * uv1ByteStride])
                                                      : glm::vec3(0.0f),

                            .color = bufferColorSet0 ? glm::make_vec4(&bufferColorSet0[v * color0ByteStride])
                                                     : glm::vec4(1.0f),

                            .tangent = bufferTangents ? glm::make_vec4(&bufferTangents[v * tangentByteStride])
                                                      : glm::vec4(0.0),
                        };

                        if (hasSkin)
                        {
                            switch (jointComponentType)
                            {
                                case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
                                    {
                                        uint16_t const* buf = static_cast<uint16_t const*>(bufferJoints);
                                        vert.joint0 = glm::uvec4(glm::make_vec4(&buf[v * jointByteStride]));
                                        break;
                                    }
                                case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
                                    {
                                        uint8_t const* buf = static_cast<uint8_t const*>(bufferJoints);
                                        vert.joint0 = glm::vec4(glm::make_vec4(&buf[v * jointByteStride]));
                                        break;
                                    }
                                default:
                                    MC_ASSERT_MSG(false,
                                                  "Joint component type {} not supported by the gltf spec",
                                                  jointComponentType);
                            }
                        }
                        else
                        {
                            vert.joint0 = glm::vec4(0.0f);
                        }

                        vert.weight0 =
                            hasSkin ? glm::make_vec4(&bufferWeights[v * weightByteStride]) : glm::vec4(0.0f);

                        // Fix for all zero weights
                        if (glm::length(vert.weight0) == 0.0f)
                        {
                            vert.weight0 = glm::vec4(1.0f, 0.0f, 0.0f, 0.0f);
                        }

                        loaderInfo.vertexPos++;
                    }
                }

                // Indices
                if (hasIndices)
                {
                    // NOTE(aether) Why not just primitive.indices?
                    tinygltf::Accessor const& accessor =
                        model.accessors[primitive.indices > -1 ? primitive.indices : 0];

                    tinygltf::BufferView const& bufferView = model.bufferViews[accessor.bufferView];
                    tinygltf::Buffer const& buffer         = model.buffers[bufferView.buffer];

                    indexCount          = static_cast<uint32_t>(accessor.count);
                    void const* dataPtr = &(buffer.data[accessor.byteOffset + bufferView.byteOffset]);

                    switch (accessor.componentType)
                    {
                        case TINYGLTF_PARAMETER_TYPE_UNSIGNED_INT:
                            {
                                uint32_t const* buf = static_cast<uint32_t const*>(dataPtr);
                                for (size_t index = 0; index < accessor.count; index++)
                                {
                                    loaderInfo.indexBuffer[loaderInfo.indexPos] = buf[index] + vertexStart;
                                    loaderInfo.indexPos++;
                                }
                                break;
                            }
                        case TINYGLTF_PARAMETER_TYPE_UNSIGNED_SHORT:
                            {
                                uint16_t const* buf = static_cast<uint16_t const*>(dataPtr);
                                for (size_t index = 0; index < accessor.count; index++)
                                {
                                    loaderInfo.indexBuffer[loaderInfo.indexPos] = buf[index] + vertexStart;
                                    loaderInfo.indexPos++;
                                }
                                break;
                            }
                        case TINYGLTF_PARAMETER_TYPE_UNSIGNED_BYTE:
                            {
                                uint8_t const* buf = static_cast<uint8_t const*>(dataPtr);
                                for (size_t index = 0; index < accessor.count; index++)
                                {
                                    loaderInfo.indexBuffer[loaderInfo.indexPos] = buf[index] + vertexStart;
                                    loaderInfo.indexPos++;
                                }
                                break;
                            }
                        default:
                            MC_ASSERT_MSG(
                                false, "Index component type {} not supported", accessor.componentType);
                    }
                }

                Primitive newPrimitive =
                    newMesh->primitives.emplace_back(indexStart,
                                                     indexCount,
                                                     vertexCount,
                                                     // Material #0 is the default material, so we add 1
                                                     primitive.material > -1 ? primitive.material + 1 : 0);

                newPrimitive.setBoundingBox(posMin, posMax);
            }

            // Mesh BB from BBs of primitives
            for (auto& p : newMesh->primitives)
            {
                if (p.bb.valid && !newMesh->bb.valid)
                {
                    newMesh->bb       = p.bb;
                    newMesh->bb.valid = true;
                }

                newMesh->bb.min = glm::min(newMesh->bb.min, p.bb.min);
                newMesh->bb.max = glm::max(newMesh->bb.max, p.bb.max);
            }
            newNode->mesh = std::move(newMesh);
        }

        if (parent)
        {
            parent->children.push_back(newNode);
        }
        else
        {
            nodes.push_back(newNode);
        }

        linearNodes.push_back(newNode);
    }

    void Model::getNodeProps(tinygltf::Node const& node,
                             tinygltf::Model const& model,
                             size_t& vertexCount,
                             size_t& indexCount)
    {
        if (node.children.size() > 0)
        {
            for (size_t i = 0; i < node.children.size(); i++)
            {
                getNodeProps(model.nodes[node.children[i]], model, vertexCount, indexCount);
            }
        }

        if (node.mesh > -1)
        {
            tinygltf::Mesh const mesh = model.meshes[node.mesh];

            for (size_t i = 0; i < mesh.primitives.size(); i++)
            {
                auto primitive = mesh.primitives[i];
                vertexCount += model.accessors[primitive.attributes.find("POSITION")->second].count;

                if (primitive.indices > -1)
                {
                    indexCount += model.accessors[primitive.indices].count;
                }
            }
        }
    }

    void Model::loadSkins(tinygltf::Model& gltfModel)
    {
        for (tinygltf::Skin& source : gltfModel.skins)
        {
            Skin* newSkin = new Skin {};
            newSkin->name = source.name;

            // Find skeleton root node
            if (source.skeleton > -1)
            {
                newSkin->skeletonRoot = nodeFromIndex(source.skeleton);
            }

            // Find joint nodes
            for (int jointIndex : source.joints)
            {
                Node* node = nodeFromIndex(jointIndex);

                if (node)
                {
                    newSkin->joints.push_back(nodeFromIndex(jointIndex));
                }
            }

            // Get inverse bind matrices from buffer
            if (source.inverseBindMatrices > -1)
            {
                tinygltf::Accessor const& accessor     = gltfModel.accessors[source.inverseBindMatrices];
                tinygltf::BufferView const& bufferView = gltfModel.bufferViews[accessor.bufferView];
                tinygltf::Buffer const& buffer         = gltfModel.buffers[bufferView.buffer];

                newSkin->inverseBindMatrices.resize(accessor.count);

                std::memcpy(newSkin->inverseBindMatrices.data(),
                            &buffer.data[accessor.byteOffset + bufferView.byteOffset],
                            accessor.count * sizeof(glm::mat4));
            }

            skins.push_back(newSkin);
        }
    }

    void Model::loadTextures(tinygltf::Model& gltfModel)
    {
        for (tinygltf::Texture& tex : gltfModel.textures)
        {
            int source = tex.source;

            // If this texture uses the KHR_texture_basisu, we need to get the source index from the extension structure
            if (tex.extensions.find("KHR_texture_basisu") != tex.extensions.end())
            {
                auto ext   = tex.extensions.find("KHR_texture_basisu");
                auto value = ext->second.Get("source");
                source     = value.Get<int>();
            }
            tinygltf::Image image = gltfModel.images[source];
            TextureSampler textureSampler;

            if (tex.sampler == -1)
            {
                textureSampler.magFilter    = vk::Filter::eLinear;
                textureSampler.minFilter    = vk::Filter::eLinear;
                textureSampler.addressModeU = vk::SamplerAddressMode::eRepeat;
                textureSampler.addressModeV = vk::SamplerAddressMode::eRepeat;
                textureSampler.addressModeW = vk::SamplerAddressMode::eRepeat;
            }
            else
            {
                textureSampler = textureSamplers[tex.sampler];
            }

            textures.push_back(
                GlTFTexture(*device, *allocator, *cmdManager, image, filePath, textureSampler));
        }
    }

    vk::SamplerAddressMode Model::getVkWrapMode(int32_t wrapMode)
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

        logger::error("Unknown wrap mode: {}", wrapMode);

        return vk::SamplerAddressMode::eRepeat;
    }

    vk::Filter Model::getVkFilterMode(int32_t filterMode)
    {
        switch (filterMode)
        {
            case -1:
            case 9728:
                return vk::Filter::eNearest;
            case 9729:
                return vk::Filter::eLinear;
            case 9984:
                return vk::Filter::eNearest;
            case 9985:
                return vk::Filter::eNearest;
            case 9986:
                return vk::Filter::eLinear;
            case 9987:
                return vk::Filter::eLinear;
        }

        logger::error("Unknown filter mode {}", filterMode);

        return vk::Filter::eNearest;
    }

    void Model::loadTextureSamplers(tinygltf::Model& gltfModel)
    {
        for (tinygltf::Sampler& smpl : gltfModel.samplers)
        {
            textureSamplers.push_back({
                .magFilter    = getVkFilterMode(smpl.magFilter),
                .minFilter    = getVkFilterMode(smpl.minFilter),
                .addressModeU = getVkWrapMode(smpl.wrapS),
                .addressModeV = getVkWrapMode(smpl.wrapT),
                .addressModeW = getVkWrapMode(smpl.wrapT),
            });
        }
    }

    void Model::loadMaterials(tinygltf::Model& gltfModel)
    {
        materials.reserve(gltfModel.materials.size() + 1);

        // Default material
        materials.push_back(Material());

        for (tinygltf::Material& mat : gltfModel.materials)
        {
            Material material {};

            material.doubleSided = mat.doubleSided;

            if (mat.values.find("baseColorTexture") != mat.values.end())
            {
                material.baseColorTexture       = &textures[mat.values["baseColorTexture"].TextureIndex()];
                material.texCoordSets.baseColor = mat.values["baseColorTexture"].TextureTexCoord();
            }

            if (mat.values.find("metallicRoughnessTexture") != mat.values.end())
            {
                material.metallicRoughnessTexture =
                    &textures[mat.values["metallicRoughnessTexture"].TextureIndex()];

                material.texCoordSets.metallicRoughness =
                    mat.values["metallicRoughnessTexture"].TextureTexCoord();
            }

            if (mat.values.find("roughnessFactor") != mat.values.end())
            {
                material.roughnessFactor = static_cast<float>(mat.values["roughnessFactor"].Factor());
            }

            if (mat.values.find("metallicFactor") != mat.values.end())
            {
                material.metallicFactor = static_cast<float>(mat.values["metallicFactor"].Factor());
            }

            if (mat.values.find("baseColorFactor") != mat.values.end())
            {
                material.baseColorFactor = glm::make_vec4(mat.values["baseColorFactor"].ColorFactor().data());
            }

            if (mat.additionalValues.find("normalTexture") != mat.additionalValues.end())
            {
                material.normalTexture = &textures[mat.additionalValues["normalTexture"].TextureIndex()];
                material.texCoordSets.normal = mat.additionalValues["normalTexture"].TextureTexCoord();
            }

            if (mat.additionalValues.find("emissiveTexture") != mat.additionalValues.end())
            {
                material.emissiveTexture = &textures[mat.additionalValues["emissiveTexture"].TextureIndex()];
                material.texCoordSets.emissive = mat.additionalValues["emissiveTexture"].TextureTexCoord();
            }

            if (mat.additionalValues.find("occlusionTexture") != mat.additionalValues.end())
            {
                material.occlusionTexture =
                    &textures[mat.additionalValues["occlusionTexture"].TextureIndex()];

                material.texCoordSets.occlusion = mat.additionalValues["occlusionTexture"].TextureTexCoord();
            }

            if (mat.additionalValues.find("alphaMode") != mat.additionalValues.end())
            {
                tinygltf::Parameter param = mat.additionalValues["alphaMode"];

                if (param.string_value == "BLEND")
                {
                    material.alphaMode = Material::ALPHAMODE_BLEND;
                }

                if (param.string_value == "MASK")
                {
                    material.alphaCutoff = 0.5f;
                    material.alphaMode   = Material::ALPHAMODE_MASK;
                }
            }

            if (mat.additionalValues.find("alphaCutoff") != mat.additionalValues.end())
            {
                material.alphaCutoff = static_cast<float>(mat.additionalValues["alphaCutoff"].Factor());
            }

            if (mat.additionalValues.find("emissiveFactor") != mat.additionalValues.end())
            {
                material.emissiveFactor = glm::vec4(
                    glm::make_vec3(mat.additionalValues["emissiveFactor"].ColorFactor().data()), 1.0);
            }

            // Extensions
            if (mat.extensions.find("KHR_materials_pbrSpecularGlossiness") != mat.extensions.end())
            {
                logger::warn("Application is not prepared to handle the specular glossiness workflow");

                auto ext = mat.extensions.find("KHR_materials_pbrSpecularGlossiness");

                if (ext->second.Has("specularGlossinessTexture"))
                {
                    auto index = ext->second.Get("specularGlossinessTexture").Get("index");
                    material.extension.specularGlossinessTexture = &textures[index.Get<int>()];

                    auto texCoordSet = ext->second.Get("specularGlossinessTexture").Get("texCoord");
                    material.texCoordSets.specularGlossiness = texCoordSet.Get<int>();
                    material.pbrWorkflows.specularGlossiness = true;
                    material.pbrWorkflows.metallicRoughness  = false;
                }

                if (ext->second.Has("diffuseTexture"))
                {
                    auto index                        = ext->second.Get("diffuseTexture").Get("index");
                    material.extension.diffuseTexture = &textures[index.Get<int>()];
                }

                if (ext->second.Has("diffuseFactor"))
                {
                    auto factor = ext->second.Get("diffuseFactor");

                    for (uint32_t i = 0; i < factor.ArrayLen(); i++)
                    {
                        auto val = factor.Get(i);
                        material.extension.diffuseFactor[i] =
                            val.IsNumber() ? (float)val.Get<double>() : (float)val.Get<int>();
                    }
                }

                if (ext->second.Has("specularFactor"))
                {
                    auto factor = ext->second.Get("specularFactor");

                    for (uint32_t i = 0; i < factor.ArrayLen(); i++)
                    {
                        auto val = factor.Get(i);
                        material.extension.specularFactor[i] =
                            val.IsNumber() ? (float)val.Get<double>() : (float)val.Get<int>();
                    }
                }
            }

            if (mat.extensions.find("KHR_materials_unlit") != mat.extensions.end())
            {
                material.unlit = true;
            }

            if (mat.extensions.find("KHR_materials_emissive_strength") != mat.extensions.end())
            {
                auto ext = mat.extensions.find("KHR_materials_emissive_strength");

                if (ext->second.Has("emissiveStrength"))
                {
                    auto value                = ext->second.Get("emissiveStrength");
                    material.emissiveStrength = (float)value.Get<double>();
                }
            }

            material.index = static_cast<uint32_t>(materials.size());
            materials.push_back(material);
        }
    }

    void Model::loadAnimations(tinygltf::Model& gltfModel)
    {
        for (tinygltf::Animation& anim : gltfModel.animations)
        {
            Animation animation { .name = anim.name };

            if (anim.name.empty())
            {
                animation.name = std::to_string(animations.size());
            }

            // Samplers
            for (auto& samp : anim.samplers)
            {
                AnimationSampler sampler {};

                if (samp.interpolation == "LINEAR")
                {
                    sampler.interpolation = AnimationSampler::InterpolationType::LINEAR;
                }

                if (samp.interpolation == "STEP")
                {
                    sampler.interpolation = AnimationSampler::InterpolationType::STEP;
                }

                if (samp.interpolation == "CUBICSPLINE")
                {
                    sampler.interpolation = AnimationSampler::InterpolationType::CUBICSPLINE;
                }

                // Read sampler input time values
                {
                    tinygltf::Accessor const& accessor     = gltfModel.accessors[samp.input];
                    tinygltf::BufferView const& bufferView = gltfModel.bufferViews[accessor.bufferView];
                    tinygltf::Buffer const& buffer         = gltfModel.buffers[bufferView.buffer];

                    assert(accessor.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT);

                    void const* dataPtr = &buffer.data[accessor.byteOffset + bufferView.byteOffset];
                    float const* buf    = static_cast<float const*>(dataPtr);

                    for (size_t index = 0; index < accessor.count; index++)
                    {
                        sampler.inputs.push_back(buf[index]);
                    }

                    for (auto input : sampler.inputs)
                    {
                        if (input < animation.start)
                        {
                            animation.start = input;
                        };
                        if (input > animation.end)
                        {
                            animation.end = input;
                        }
                    }
                }

                // Read sampler output T/R/S values
                {
                    tinygltf::Accessor const& accessor     = gltfModel.accessors[samp.output];
                    tinygltf::BufferView const& bufferView = gltfModel.bufferViews[accessor.bufferView];
                    tinygltf::Buffer const& buffer         = gltfModel.buffers[bufferView.buffer];

                    assert(accessor.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT);

                    void const* dataPtr = &buffer.data[accessor.byteOffset + bufferView.byteOffset];

                    switch (accessor.type)
                    {
                        case TINYGLTF_TYPE_VEC3:
                            {
                                glm::vec3 const* buf = static_cast<glm::vec3 const*>(dataPtr);
                                for (size_t index = 0; index < accessor.count; index++)
                                {
                                    sampler.outputsVec4.push_back(glm::vec4(buf[index], 0.0f));
                                    sampler.outputs.push_back(buf[index][0]);
                                    sampler.outputs.push_back(buf[index][1]);
                                    sampler.outputs.push_back(buf[index][2]);
                                }
                                break;
                            }
                        case TINYGLTF_TYPE_VEC4:
                            {
                                glm::vec4 const* buf = static_cast<glm::vec4 const*>(dataPtr);
                                for (size_t index = 0; index < accessor.count; index++)
                                {
                                    sampler.outputsVec4.push_back(buf[index]);
                                    sampler.outputs.push_back(buf[index][0]);
                                    sampler.outputs.push_back(buf[index][1]);
                                    sampler.outputs.push_back(buf[index][2]);
                                    sampler.outputs.push_back(buf[index][3]);
                                }
                                break;
                            }
                        default:
                            {
                                MC_ASSERT_MSG(false, "Unknown type");
                                break;
                            }
                    }
                }

                animation.samplers.push_back(sampler);
            }

            // Channels
            for (auto& source : anim.channels)
            {
                AnimationChannel channel {};

                if (source.target_path == "weights")
                {
                    logger::warn("weights not yet supported, skipping channel");
                    continue;
                }

                if (source.target_path == "rotation")
                {
                    channel.path = AnimationChannel::PathType::ROTATION;
                }

                if (source.target_path == "translation")
                {
                    channel.path = AnimationChannel::PathType::TRANSLATION;
                }

                if (source.target_path == "scale")
                {
                    channel.path = AnimationChannel::PathType::SCALE;
                }

                channel.samplerIndex = source.sampler;
                channel.node         = nodeFromIndex(source.target_node);

                if (!channel.node)
                {
                    continue;
                }

                animation.channels.push_back(channel);
            }

            animations.push_back(animation);
        }
    }

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
                logger::info("Model uses KHR_texture_basisu, initializing basisu transcoder");
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
            // Initial pose
            if (node->mesh)
            {
                node->update();
            }
        }

        size_t vertexBufferSize = vertexCount * sizeof(Vertex);
        size_t indexBufferSize  = indexCount * sizeof(uint32_t);

        MC_ASSERT(vertexBufferSize > 0);

        ScopedCommandBuffer cmdBuf(
            *device, cmdManager->getTransferCmdPool(), device->getTransferQueue(), true);

        GPUBuffer vertexStaging(*device,
                                *allocator,
                                "Vertex staging",
                                vertexBufferSize,
                                vk::BufferUsageFlagBits::eTransferSrc,
                                VMA_MEMORY_USAGE_AUTO,
                                VMA_ALLOCATION_CREATE_MAPPED_BIT |
                                    VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);

        std::memcpy(vertexStaging.getMappedData(), loaderInfo.vertexBuffer, vertexBufferSize);

        vertices =
            GPUBuffer(*device,
                      *allocator,
                      "Main vertex buffer",
                      vertexBufferSize,
                      vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eShaderDeviceAddress,
                      VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
                      VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT);

        vertexBufferAddress =
            device->get().getBufferAddress(vk::BufferDeviceAddressInfo().setBuffer(vertices));

        cmdBuf->copyBuffer(vertexStaging, vertices, vk::BufferCopy().setSize(vertexBufferSize));

        if (indexBufferSize > 0)
        {
            GPUBuffer indexStaging(*device,
                                   *allocator,
                                   "Index staging",
                                   indexBufferSize,
                                   vk::BufferUsageFlagBits::eTransferSrc,
                                   VMA_MEMORY_USAGE_AUTO,
                                   VMA_ALLOCATION_CREATE_MAPPED_BIT |
                                       VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);

            std::memcpy(indexStaging.getMappedData(), loaderInfo.indexBuffer, indexBufferSize);

            indices = GPUBuffer(*device,
                                *allocator,
                                "Main index buffer",
                                indexBufferSize,
                                vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eIndexBuffer,
                                VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
                                VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT);

            cmdBuf->copyBuffer(indexStaging, indices, vk::BufferCopy().setSize(indexBufferSize));
        }

        cmdBuf.flush();

        delete[] loaderInfo.vertexBuffer;
        delete[] loaderInfo.indexBuffer;

        getSceneDimensions();

        createMaterialBuffer();
        setupDescriptors();
    }

    void Model::drawNode(Node* node, vk::CommandBuffer commandBuffer)
    {
        if (node->mesh)
        {
            for (Primitive& primitive : node->mesh->primitives)
            {
                commandBuffer.drawIndexed(primitive.indexCount, 1, primitive.firstIndex, 0, 0);
            }
        }

        for (auto& child : node->children)
        {
            drawNode(child, commandBuffer);
        }
    }

    void Model::draw(vk::CommandBuffer commandBuffer)
    {
        commandBuffer.bindIndexBuffer(indices, 0, vk::IndexType::eUint32);

        for (auto& node : nodes)
        {
            drawNode(node, commandBuffer);
        }
    }

    void Model::calculateBoundingBox(Node* node, Node* parent)
    {
        BoundingBox parentBvh = parent ? parent->bvh : BoundingBox(dimensions.min, dimensions.max);

        if (node->mesh)
        {
            if (node->mesh->bb.valid)
            {
                node->aabb = node->mesh->bb.getAABB(node->getMatrix());
                if (node->children.size() == 0)
                {
                    node->bvh.min   = node->aabb.min;
                    node->bvh.max   = node->aabb.max;
                    node->bvh.valid = true;
                }
            }
        }

        parentBvh.min = glm::min(parentBvh.min, node->bvh.min);
        parentBvh.max = glm::min(parentBvh.max, node->bvh.max);

        for (auto& child : node->children)
        {
            calculateBoundingBox(child, node);
        }
    }

    void Model::getSceneDimensions()
    {
        // Calculate binary volume hierarchy for all nodes in the scene
        for (auto node : linearNodes)
        {
            calculateBoundingBox(node, nullptr);
        }

        dimensions.min = glm::vec3(std::numeric_limits<float>::max());
        dimensions.max = glm::vec3(-std::numeric_limits<float>::max());

        for (auto node : linearNodes)
        {
            if (node->bvh.valid)
            {
                dimensions.min = glm::min(dimensions.min, node->bvh.min);
                dimensions.max = glm::max(dimensions.max, node->bvh.max);
            }
        }

        // Calculate scene aabb
        aabb       = glm::scale(glm::mat4(1.0f),
                          glm::vec3(dimensions.max[0] - dimensions.min[0],
                                    dimensions.max[1] - dimensions.min[1],
                                    dimensions.max[2] - dimensions.min[2]));
        aabb[3][0] = dimensions.min[0];
        aabb[3][1] = dimensions.min[1];
        aabb[3][2] = dimensions.min[2];
    }

    void Model::updateAnimation(uint32_t index, float time)
    {
        if (animations.empty())
        {
            logger::warn("glTF does not contain animation");
            return;
        }

        if (index > static_cast<uint32_t>(animations.size()) - 1)
        {
            logger::warn("No animation with index {}", index);
            return;
        }

        Animation& animation = animations[index];

        bool updated = false;

        for (auto& channel : animation.channels)
        {
            AnimationSampler& sampler = animation.samplers[channel.samplerIndex];

            if (sampler.inputs.size() > sampler.outputsVec4.size())
            {
                continue;
            }

            for (size_t i = 0; i < sampler.inputs.size() - 1; i++)
            {
                if ((time >= sampler.inputs[i]) && (time <= sampler.inputs[i + 1]))
                {
                    float u = std::max(0.0f, time - sampler.inputs[i]) /
                              (sampler.inputs[i + 1] - sampler.inputs[i]);

                    if (u <= 1.0f)
                    {
                        switch (channel.path)
                        {
                            case AnimationChannel::PathType::TRANSLATION:
                                sampler.translate(i, time, channel.node);
                                break;
                            case AnimationChannel::PathType::SCALE:
                                sampler.scale(i, time, channel.node);
                                break;
                            case AnimationChannel::PathType::ROTATION:
                                sampler.rotate(i, time, channel.node);
                                break;
                        }

                        updated = true;
                    }
                }
            }
        }

        if (updated)
        {
            for (auto& node : nodes)
            {
                node->update();
            }
        }
    }

    Node* Model::findNode(Node* parent, uint32_t index)
    {
        Node* nodeFound = nullptr;

        if (parent->index == index)
        {
            return parent;
        }

        for (auto& child : parent->children)
        {
            nodeFound = findNode(child, index);

            if (nodeFound)
            {
                break;
            }
        }

        return nodeFound;
    }

    Node* Model::nodeFromIndex(uint32_t index)
    {
        Node* nodeFound = nullptr;

        for (auto& node : nodes)
        {
            nodeFound = findNode(node, index);

            if (nodeFound)
            {
                break;
            }
        }
        return nodeFound;
    }

    void Model::createMaterialBuffer()
    {
        std::vector<ShaderMaterial> shaderMaterials {};

        for (auto& material : materials)
        {
            ShaderMaterial shaderMaterial {};

            shaderMaterial.emissiveFactor   = material.emissiveFactor;
            shaderMaterial.emissiveStrength = material.emissiveStrength;

            // To save space, availabilty and texture coordinate set are combined
            // -1 = texture not used for this material, >= 0 texture used and index of
            // texture coordinate set

            shaderMaterial.colorTextureSet =
                material.baseColorTexture != nullptr ? material.texCoordSets.baseColor : -1;

            shaderMaterial.normalTextureSet =
                material.normalTexture != nullptr ? material.texCoordSets.normal : -1;

            shaderMaterial.occlusionTextureSet =
                material.occlusionTexture != nullptr ? material.texCoordSets.occlusion : -1;

            shaderMaterial.emissiveTextureSet =
                material.emissiveTexture != nullptr ? material.texCoordSets.emissive : -1;

            shaderMaterial.alphaMask = static_cast<float>(material.alphaMode == Material::ALPHAMODE_MASK);
            shaderMaterial.alphaMaskCutoff = material.alphaCutoff;

            if (material.pbrWorkflows.metallicRoughness)
            {
                // Metallic roughness workflow
                shaderMaterial.workflow        = static_cast<float>(PBRWorkflows::MetallicRoughness);
                shaderMaterial.baseColorFactor = material.baseColorFactor;
                shaderMaterial.metallicFactor  = material.metallicFactor;
                shaderMaterial.roughnessFactor = material.roughnessFactor;
                shaderMaterial.physicalDescriptorTextureSet = material.metallicRoughnessTexture != nullptr
                                                                  ? material.texCoordSets.metallicRoughness
                                                                  : -1;
                shaderMaterial.colorTextureSet =
                    material.baseColorTexture != nullptr ? material.texCoordSets.baseColor : -1;
            }
            else
            {
                if (material.pbrWorkflows.specularGlossiness)
                {
                    // Specular glossiness workflow
                    shaderMaterial.workflow = static_cast<float>(PBRWorkflows::SpecularGlossiness);
                    shaderMaterial.physicalDescriptorTextureSet =
                        material.extension.specularGlossinessTexture != nullptr
                            ? material.texCoordSets.specularGlossiness
                            : -1;
                    shaderMaterial.colorTextureSet =
                        material.extension.diffuseTexture != nullptr ? material.texCoordSets.baseColor : -1;
                    shaderMaterial.diffuseFactor  = material.extension.diffuseFactor;
                    shaderMaterial.specularFactor = glm::vec4(material.extension.specularFactor, 1.0f);
                }
            }

            shaderMaterials.push_back(shaderMaterial);
        }

        vk::DeviceSize bufferSize = shaderMaterials.size() * sizeof(ShaderMaterial);

        GPUBuffer stagingBuffer(*allocator,
                                bufferSize,
                                vk::BufferUsageFlagBits::eTransferSrc,
                                VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                                VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                                    VMA_ALLOCATION_CREATE_MAPPED_BIT);

        std::memcpy(stagingBuffer.getMappedData(), shaderMaterials.data(), bufferSize);

        materialBuffer =
            GPUBuffer(*allocator,
                      bufferSize,
                      vk::BufferUsageFlagBits::eShaderDeviceAddress | vk::BufferUsageFlagBits::eTransferDst,
                      VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
                      VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT);

        materialBufferAddress =
            device->get().getBufferAddress(vk::BufferDeviceAddressInfo().setBuffer(materialBuffer));

        // TODO(aether) the deconstructor will block until the copy is over
        // not the most performant approach
        ScopedCommandBuffer(*device, cmdManager->getTransferCmdPool(), device->getTransferQueue(), true)
            ->copyBuffer(stagingBuffer, materialBuffer, vk::BufferCopy().setSize(bufferSize));
    }

    void Model::setupDescriptors()
    {
        int imageSamplerCount = materials.size() * 5;

        std::vector<DescriptorAllocator::PoolSizeRatio> sizes {
            { vk::DescriptorType::eCombinedImageSampler,
             // TODO(aether)                           vvvvvvvvvvvvvvvvvvvv why?
              static_cast<float>(imageSamplerCount) /* * kNumFramesInFlight */ }
        };

        m_materialDescriptorAllocator = DescriptorAllocator(*device, materials.size(), sizes);

        // Per-Material descriptor sets
        for (auto& material : materials)
        {
            material.descriptorSet =
                m_materialDescriptorAllocator.allocate(*device, m_materialDescriptorSetLayout);

            DescriptorWriter writer;

            // Default the diffuse and metallicRoughness textures initially
            for (int i = 0; i < 2; ++i)
            {
                writer.write_image(i,
                                   m_dummyImage,
                                   m_dummySampler,
                                   vk::ImageLayout::eShaderReadOnlyOptimal,
                                   vk::DescriptorType::eCombinedImageSampler);
            }

            std::array texturesToWrite {
                static_cast<GlTFTexture*>(nullptr),
                static_cast<GlTFTexture*>(nullptr),
                material.occlusionTexture,
                material.emissiveTexture,
                material.normalTexture,
            };

            if (material.pbrWorkflows.metallicRoughness)
            {
                texturesToWrite[0] = material.baseColorTexture;
                texturesToWrite[1] = material.metallicRoughnessTexture;
            }
            else
            {
                texturesToWrite[0] = material.extension.diffuseTexture;
                texturesToWrite[1] = material.extension.specularGlossinessTexture;
            }

            for (auto [i, tex] : vi::enumerate(texturesToWrite))
            {
                if (!tex)
                {
                    writer.write_image(i,
                                       m_dummyImage,
                                       m_dummySampler,
                                       vk::ImageLayout::eShaderReadOnlyOptimal,
                                       vk::DescriptorType::eCombinedImageSampler);

                    continue;
                }

                if constexpr (kDebug)
                {
                    std::string_view type;

                    switch (i)
                    {
                        case 0:
                            type = "diffuse";
                            break;
                        case 1:
                            type = "metallic/roughness";
                            break;
                        case 2:
                            type = "occlusion";
                            break;
                        case 3:
                            type = "emissive";
                            break;
                        case 4:
                            type = "normal";
                            break;
                    }

                    // TODO(aether) add more verbosity to the name of other buffers and images just like here
                    tex->image.setName(std::format(
                        "{} (Material #{} {} texture)", tex->image.getName(), material.index, type));
                }

                writer.write_image(i,
                                   tex->image.getImageView(),
                                   tex->sampler,
                                   vk::ImageLayout::eShaderReadOnlyOptimal,
                                   vk::DescriptorType::eCombinedImageSampler);
            }

            writer.update_set(*device, material.descriptorSet);
        }
    }
}  // namespace renderer::backend
