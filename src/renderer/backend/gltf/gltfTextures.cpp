#include <mc/renderer/backend/gltf/gltfTextures.hpp>
#include <mc/renderer/backend/gltf/loader.hpp>

#include <fstream>
#include <ranges>

#include "basisu_transcoder.h"

namespace vi = std::ranges::views;

namespace renderer::backend
{
    // Loads the image for this texture. Supports both glTF's web formats (jpg, png, embedded and external files) as well as external KTX2 files with basis universal texture compression
    GlTFTexture::GlTFTexture(Device& device,
                             CommandManager& cmdManager,
                             ResourceManager<GPUBuffer>& bufferManager,
                             ResourceManager<Image>& imageManager,
                             tinygltf::Image& gltfimage,
                             std::filesystem::path path,
                             TextureSampler textureSampler)
        : m_device { &device }, m_commandManager { &cmdManager }
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

            auto stagingBuffer = bufferManager.create("Image staging buffer (compressed)",
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

            // FIXME(aether) stop using imageManager here
            // differ all this processing to the Texture class
            texture = imageManager.create(std::format("Compressed gltf texture ({})", gltfimage.uri),
                                          vk::Extent2D { width, height },
                                          format,
                                          vk::SampleCountFlagBits::e1,
                                          vk::ImageUsageFlagBits::eTransferSrc |
                                              vk::ImageUsageFlagBits::eTransferDst |
                                              vk::ImageUsageFlagBits::eSampled,
                                          vk::ImageAspectFlagBits::eColor,
                                          mipLevels);

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
                .image            = texture,
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
                    stagingBuffer, texture, vk::ImageLayout::eTransferDstOptimal, { bufferCopyRegion });

                bufferOffset += outputSize;
            }

            imageMemoryBarrier.oldLayout        = vk::ImageLayout::eTransferDstOptimal;
            imageMemoryBarrier.newLayout        = vk::ImageLayout::eShaderReadOnlyOptimal;
            imageMemoryBarrier.srcAccessMask    = vk::AccessFlagBits::eTransferWrite;
            imageMemoryBarrier.dstAccessMask    = vk::AccessFlagBits::eShaderRead;
            imageMemoryBarrier.image            = texture;
            imageMemoryBarrier.subresourceRange = subresourceRange;

            copyCmd->pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
                                     vk::PipelineStageFlagBits::eAllCommands,
                                     {},
                                     {},
                                     {},
                                     { imageMemoryBarrier });

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

            auto stagingBuffer =
                bufferManager.create("Image staging buffer (uncompressed)",
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

            texture = imageManager.create(std::format("Uncompressed gltf texture ({})", gltfimage.uri),
                                          vk::Extent2D { width, height },
                                          format,
                                          vk::SampleCountFlagBits::e1,
                                          vk::ImageUsageFlagBits::eTransferSrc |
                                              vk::ImageUsageFlagBits::eTransferDst |
                                              vk::ImageUsageFlagBits::eSampled,
                                          vk::ImageAspectFlagBits::eColor,
                                          mipLevels);

            ScopedCommandBuffer cmdBuf(device, cmdManager.getMainCmdPool(), device.getMainQueue(), true);

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
                    .image            = texture,
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
                stagingBuffer, texture, vk::ImageLayout::eTransferDstOptimal, { bufferCopyRegion });

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
                                        .image            = texture,
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
                                            .image            = texture,
                                            .subresourceRange = mipSubRange,
                                        } });

                cmdBuf->blitImage(texture,
                                  vk::ImageLayout::eTransferSrcOptimal,
                                  texture,
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
                                            .image            = texture,
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
                                        .image            = texture,
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

    void Model::setupDescriptors()
    {
        MC_ASSERT(materials.size() <= kMaxBindlessResources);

        std::vector<DescriptorAllocator::PoolSizeRatio> sizes {
            { vk::DescriptorType::eCombinedImageSampler, 5 }
        };

        m_materialDescriptorAllocator = DescriptorAllocator(
            *m_device, materials.size(), sizes, vk::DescriptorPoolCreateFlagBits::eUpdateAfterBindEXT);

        bindlessMaterialDescriptorSet =
            m_materialDescriptorAllocator.allocate(*m_device, m_materialDescriptorSetLayout);

        std::vector<vk::DescriptorImageInfo> imageInfos(materials.size() * 5);

        // Per-Material descriptor sets
        for (auto [materialIndex, material] : vi::enumerate(materials))
        {
            std::array texturesToWrite {
                static_cast<GlTFTexture*>(nullptr),
                static_cast<GlTFTexture*>(nullptr),
                material.occlusionTexture,
                material.emissiveTexture,
                material.normalTexture,
            };

            if (material.pbrWorkflow == PBRWorkflows::metallicRoughness)
            {
                texturesToWrite[0] = material.baseColorTexture;
                texturesToWrite[1] = material.metallicRoughnessTexture;
            }
            else
            {
                texturesToWrite[0] = material.extension.diffuseTexture;
                texturesToWrite[1] = material.extension.specularGlossinessTexture;
            }

            for (auto [texIndex, tex] : vi::enumerate(texturesToWrite))
            {
                if (!tex)
                {
                    imageInfos[(materialIndex * 5) + texIndex] = {
                        .sampler     = m_dummySampler,
                        .imageView   = m_dummyImage,
                        .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
                    };
                    continue;
                }

                ResourceAccessor<Image>& img = tex->texture;

                if constexpr (kDebug)
                {
                    std::string_view type;

                    switch (texIndex)
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
                    img.setName(
                        std::format("{} (Material #{} {} texture)", img.getName(), material.index, type));
                }

                imageInfos[(materialIndex * 5) + texIndex] = {
                    .sampler     = tex->sampler,
                    .imageView   = img.getImageView(),
                    .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
                };
            }
        }

        DescriptorWriter()
            .writeImages(0,
                         vk::ImageLayout::eShaderReadOnlyOptimal,
                         vk::DescriptorType::eCombinedImageSampler,
                         imageInfos)
            .updateSet(*m_device, bindlessMaterialDescriptorSet);
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

            textures.push_back(GlTFTexture(*m_device,
                                           *m_cmdManager,
                                           *m_bufferManager,
                                           *m_imageManager,
                                           image,
                                           filePath,
                                           textureSampler));
        }
    }

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
}  // namespace renderer::backend
