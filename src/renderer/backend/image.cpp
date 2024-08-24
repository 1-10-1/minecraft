#include <mc/asserts.hpp>
#include <mc/exceptions.hpp>
#include <mc/renderer/backend/allocator.hpp>
#include <mc/renderer/backend/buffer.hpp>
#include <mc/renderer/backend/command.hpp>
#include <mc/renderer/backend/image.hpp>
#include <mc/renderer/backend/vk_checker.hpp>

#include <stb_image.h>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_core.h>
#include <vulkan/vulkan_structs.hpp>

namespace
{
    using namespace renderer::backend;

    // Does not follow vk::Guide, used only in the old tutorial's texture creation code
    // Dont use this
    [[deprecated("Used in the old tutorial")]] void transitionImageLayout(ScopedCommandBuffer& commandBuffer,
                                                                          vk::Image image,
                                                                          vk::Format format,
                                                                          vk::ImageLayout oldLayout,
                                                                          vk::ImageLayout newLayout,
                                                                          uint32_t mipLevels);
}  // namespace

namespace renderer::backend
{
    StbiImage::StbiImage(std::string_view const& path)
    {
        int texWidth { 0 };
        int texHeight { 0 };
        int texChannels { 0 };

        // TODO(aether) instead of memcpying to the mapped gpu buffer region, make this
        // directly load it into the mapped region.
        m_data = stbi_load(path.data(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);

        m_dimensions = vk::Extent2D { static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight) };

        m_size = static_cast<size_t>(texWidth) * texHeight * 4;  // 4 channels, RGBA

        MC_ASSERT_MSG(m_data, "Failed to load texture");
    }

    StbiImage::~StbiImage()
    {
        stbi_image_free(m_data);
    }

    BasicImage::BasicImage(Device const& device,
                           Allocator const& allocator,
                           vk::Extent2D dimensions,
                           vk::Format format,
                           vk::SampleCountFlagBits sampleCount,
                           vk::ImageUsageFlags usageFlags,
                           vk::ImageAspectFlags aspectFlags,
                           uint32_t mipLevels,
                           std::string_view name)
        : m_device { &device },
          m_allocator { &allocator },
          m_format { format },
          m_sampleCount { sampleCount },
          m_usageFlags { usageFlags },
          m_aspectFlags { aspectFlags },
          m_mipLevels { mipLevels },
          m_dimensions { dimensions }
    {
        create();

        if (!name.empty())
        {
            setName(name);
        }
    }

    BasicImage::~BasicImage()
    {
        destroy();
    };

    void BasicImage::create()
    {
        createImage(m_format,
                    vk::ImageTiling::eOptimal,
                    m_usageFlags,
                    vk::MemoryPropertyFlagBits::eDeviceLocal,
                    m_mipLevels,
                    m_sampleCount);

        // If the image is solely being used for transfer, dont make a view
        if ((m_usageFlags & (vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst)) <
            m_usageFlags)
        {
            createImageView(m_format, m_aspectFlags, 1);
        }
    }

    void BasicImage::destroy()
    {
        if (!m_imageHandle)
        {
            return;
        }

        m_imageView.clear();
        vmaDestroyImage(*m_allocator, m_imageHandle, m_allocation);

        m_imageHandle = nullptr;
    }

    void
    BasicImage::copyTo(vk::CommandBuffer cmdBuf, vk::Image dst, vk::Extent2D dstSize, vk::Extent2D offset)
    {
        vk::ImageBlit2 blitRegion {};

        blitRegion.srcOffsets[1].x = static_cast<int32_t>(offset.width);
        blitRegion.srcOffsets[1].y = static_cast<int32_t>(offset.height);
        blitRegion.srcOffsets[1].z = 1;

        blitRegion.dstOffsets[1].x = static_cast<int32_t>(dstSize.width);
        blitRegion.dstOffsets[1].y = static_cast<int32_t>(dstSize.height);
        blitRegion.dstOffsets[1].z = 1;

        blitRegion.srcSubresource.aspectMask     = vk::ImageAspectFlagBits::eColor;
        blitRegion.srcSubresource.baseArrayLayer = 0;
        blitRegion.srcSubresource.layerCount     = 1;
        blitRegion.srcSubresource.mipLevel       = 0;

        blitRegion.dstSubresource.aspectMask     = vk::ImageAspectFlagBits::eColor;
        blitRegion.dstSubresource.baseArrayLayer = 0;
        blitRegion.dstSubresource.layerCount     = 1;
        blitRegion.dstSubresource.mipLevel       = 0;

        vk::BlitImageInfo2 blitInfo {};

        blitInfo.dstImage       = dst;
        blitInfo.dstImageLayout = vk::ImageLayout::eTransferDstOptimal;

        blitInfo.srcImage       = m_imageHandle;
        blitInfo.srcImageLayout = vk::ImageLayout::eTransferSrcOptimal;

        blitInfo.filter      = vk::Filter::eLinear;
        blitInfo.regionCount = 1;
        blitInfo.pRegions    = &blitRegion;

        cmdBuf.blitImage2(blitInfo);
    };

    void BasicImage::transition(vk::CommandBuffer cmdBuf,
                                vk::Image image,
                                vk::ImageLayout currentLayout,
                                vk::ImageLayout newLayout)
    {
        // TODO(aether) Those stage masks will cause the pipeline to stall
        // figure out the appropriate stages based on a new parameter or the ones already given
        vk::ImageMemoryBarrier2 imageBarrier {
            .srcStageMask     = vk::PipelineStageFlagBits2::eAllCommands,
            .srcAccessMask    = vk::AccessFlagBits2::eMemoryWrite,
            .dstStageMask     = vk::PipelineStageFlagBits2::eAllCommands,
            .dstAccessMask    = vk::AccessFlagBits2::eMemoryWrite | vk::AccessFlagBits2::eMemoryRead,
            .oldLayout        = currentLayout,
            .newLayout        = newLayout,
            .image            = image,
            .subresourceRange = {
                .aspectMask     = (newLayout == vk::ImageLayout::eDepthAttachmentOptimal)
                                  ? vk::ImageAspectFlagBits::eDepth : vk::ImageAspectFlagBits::eColor,
                .baseMipLevel   = 0,
                .levelCount     = vk::RemainingMipLevels,
                .baseArrayLayer = 0,
                .layerCount     = vk::RemainingArrayLayers,
            },

        };

        vk::DependencyInfo depInfo {
            .imageMemoryBarrierCount = 1,
            .pImageMemoryBarriers    = &imageBarrier,
        };

        cmdBuf.pipelineBarrier2(depInfo);
    };

    void BasicImage::createImage(vk::Format format,
                                 vk::ImageTiling tiling,
                                 vk::ImageUsageFlags usage,
                                 vk::MemoryPropertyFlags properties,
                                 uint32_t mipLevels,
                                 vk::SampleCountFlagBits numSamples)
    {
        vk::ImageCreateInfo imageInfo {
            .imageType     = vk::ImageType::e2D,
            .format        = format,
            .extent        = { m_dimensions.width, m_dimensions.height, 1 },
            .mipLevels     = mipLevels,
            .arrayLayers   = 1,
            .samples       = numSamples,
            .tiling        = tiling,
            .usage         = usage,
            .sharingMode   = vk::SharingMode::eExclusive,
            .initialLayout = vk::ImageLayout::eUndefined,
        };

        VmaAllocationCreateInfo imageAllocInfo = {
            .usage         = VMA_MEMORY_USAGE_GPU_ONLY,
            .requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        };

        vmaCreateImage(*m_allocator,
                       &static_cast<VkImageCreateInfo&>(imageInfo),
                       &imageAllocInfo,
                       &m_imageHandle,
                       &m_allocation,
                       nullptr);
    }

    void BasicImage::createImageView(vk::Format format, vk::ImageAspectFlags aspectFlags, uint32_t mipLevels)
    {
        vk::ImageViewCreateInfo viewInfo {
            .image              = m_imageHandle,
            .viewType           = vk::ImageViewType::e2D,
            .format             = format,
            .subresourceRange   = {
                .aspectMask     = aspectFlags,
                .baseMipLevel   = 0,
                .levelCount     = mipLevels,
                .baseArrayLayer = 0,
                .layerCount     = 1,
            }
        };

        m_imageView = m_device->get().createImageView(viewInfo) >> ResultChecker();
    }

    Image::Image(Device& device,
                 Allocator& allocator,
                 CommandManager& commandManager,
                 StbiImage const& stbiImage)
        : m_device { &device },
          m_allocator { &allocator },
          m_commandManager { &commandManager },
          m_image { *m_device,
                    allocator,
                    stbiImage.getDimensions(),
                    vk::Format::eR8G8B8A8Unorm,
                    vk::SampleCountFlagBits::e1,
                    vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst |
                        vk::ImageUsageFlagBits::eSampled,
                    vk::ImageAspectFlagBits::eColor,
                    static_cast<uint32_t>(std::floor(std::log2(
                        std::max(stbiImage.getDimensions().width, stbiImage.getDimensions().height)))) +
                        1 }
    {
        vk::Extent2D dimensions = stbiImage.getDimensions();

        uint32_t mipLevels = m_image.getMipLevels();

        GPUBuffer uploadBuffer(*m_allocator,
                               stbiImage.getDataSize(),
                               vk::BufferUsageFlagBits::eTransferSrc,
                               VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                               VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                                   VMA_ALLOCATION_CREATE_MAPPED_BIT);

        std::memcpy(uploadBuffer.getMappedData(), stbiImage.getData(), stbiImage.getDataSize());

        {
            ScopedCommandBuffer commandBuffer(
                *m_device, m_commandManager->getMainCmdPool(), m_device->getMainQueue());

            BasicImage::transition(
                commandBuffer, m_image, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal);

            transitionImageLayout(commandBuffer,
                                  m_image,
                                  vk::Format::eR8G8B8A8Unorm,
                                  vk::ImageLayout::eUndefined,
                                  vk::ImageLayout::eTransferDstOptimal,
                                  mipLevels);

            vk::BufferImageCopy region {
                .bufferOffset      = 0,
                .bufferRowLength   = 0,
                .bufferImageHeight = 0,
                .imageSubresource  = {
                    .aspectMask     = vk::ImageAspectFlagBits::eColor,
                    .mipLevel       = 0,
                    .baseArrayLayer = 0,
                    .layerCount     = 1,
                },
                .imageOffset = { 0, 0, 0 },
                .imageExtent = { dimensions.width, dimensions.height, 1 },
            };

            commandBuffer->copyBufferToImage(
                uploadBuffer, m_image, vk::ImageLayout::eTransferDstOptimal, { region });

            //transitioned to VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL here
            // generateMipmaps(commandBuffer,
            //                 m_image,
            //                 { dimensions.width, dimensions.height },
            //                 vk::Format::eR8G8B8A8Unorm,
            //                 mipLevels);

            commandBuffer->pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                                           vk::PipelineStageFlagBits::eFragmentShader,
                                           vk::DependencyFlags { 0 },
                                           {},
                                           {},
                                           {
                {
                    .srcAccessMask = vk::AccessFlagBits::eTransferWrite,
                    .dstAccessMask = vk::AccessFlagBits::eShaderRead,
                    .oldLayout     = vk::ImageLayout::eTransferDstOptimal,
                    .newLayout     = vk::ImageLayout::eShaderReadOnlyOptimal,
                    .srcQueueFamilyIndex = vk::QueueFamilyIgnored,
                    .dstQueueFamilyIndex = vk::QueueFamilyIgnored,
                    .image = m_image,
                    .subresourceRange = {
                        .aspectMask     = vk::ImageAspectFlagBits::eColor,
                        .levelCount     = 1,
                        .baseArrayLayer = 0,
                        .layerCount     = 1,
                    },
                }
            });
        }
    }

    Image::Image(Device& device,
                 Allocator& allocator,
                 CommandManager& commandManager,
                 vk::Extent2D dimensions,
                 void* data,
                 size_t dataSize)
        : m_device { &device },
          m_allocator { &allocator },
          m_commandManager { &commandManager },
          m_image {
              *m_device,
              allocator,
              dimensions,
              vk::Format::eR8G8B8A8Unorm,
              vk::SampleCountFlagBits::e1,
              vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst |
                  vk::ImageUsageFlagBits::eSampled,
              vk::ImageAspectFlagBits::eColor,
              static_cast<uint32_t>(std::floor(std::log2(std::max(dimensions.width, dimensions.height)))) + 1
          }
    {
        uint32_t mipLevels = m_image.getMipLevels();

        GPUBuffer uploadBuffer(*m_allocator,
                               dataSize,
                               vk::BufferUsageFlagBits::eTransferSrc,
                               VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                               VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                                   VMA_ALLOCATION_CREATE_MAPPED_BIT);

        std::memcpy(uploadBuffer.getMappedData(), data, dataSize);

        {
            // TODO(aether) graphics or transfer?
            ScopedCommandBuffer commandBuffer(
                *m_device, m_commandManager->getMainCmdPool(), m_device->getMainQueue());

            BasicImage::transition(
                commandBuffer, m_image, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal);

            transitionImageLayout(commandBuffer,
                                  m_image,
                                  vk::Format::eR8G8B8A8Unorm,
                                  vk::ImageLayout::eUndefined,
                                  vk::ImageLayout::eTransferDstOptimal,
                                  mipLevels);

            vk::BufferImageCopy region {
                .bufferOffset      = 0,
                .bufferRowLength   = 0,
                .bufferImageHeight = 0,
                .imageSubresource  = {
                    .aspectMask     = vk::ImageAspectFlagBits::eColor,
                    .mipLevel       = 0,
                    .baseArrayLayer = 0,
                    .layerCount     = 1,
                },
                .imageOffset = { 0, 0, 0 },
                .imageExtent = { dimensions.width, dimensions.height, 1 },
            };

            commandBuffer->copyBufferToImage(
                uploadBuffer, m_image, vk::ImageLayout::eTransferDstOptimal, { region });

            //transitioned to VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL here
            // generateMipmaps(commandBuffer,
            //                 m_image,
            //                 { dimensions.width, dimensions.height },
            //                 vk::Format::eR8G8B8A8Unorm,
            //                 mipLevels);

            commandBuffer->pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                                           vk::PipelineStageFlagBits::eFragmentShader,
                                           vk::DependencyFlags { 0 },
                                           {},
                                           {},
                                           {
                {
                    .srcAccessMask = vk::AccessFlagBits::eTransferWrite,
                    .dstAccessMask = vk::AccessFlagBits::eShaderRead,
                    .oldLayout     = vk::ImageLayout::eTransferDstOptimal,
                    .newLayout     = vk::ImageLayout::eShaderReadOnlyOptimal,
                    .srcQueueFamilyIndex = vk::QueueFamilyIgnored,
                    .dstQueueFamilyIndex = vk::QueueFamilyIgnored,
                    .image = m_image,
                    .subresourceRange = {
                        .aspectMask     = vk::ImageAspectFlagBits::eColor,
                        .levelCount     = 1,
                        .baseArrayLayer = 0,
                        .layerCount     = 1,
                    },
                }
            });
        }
    }

    void generateMipmaps(ScopedCommandBuffer& commandBuffer,
                         vk::Image image,
                         vk::Extent2D dimensions,
                         vk::Format imageFormat,
                         uint32_t mipLevels)
    {
        // MC_ASSERT(m_device->getFormatProperties(imageFormat).optimalTilingFeatures &
        //           vk::FormatFeatureFlagBits::eSampledImageFilterLinear);

        vk::ImageMemoryBarrier barrier {
                .srcQueueFamilyIndex = vk::QueueFamilyIgnored,
                .dstQueueFamilyIndex = vk::QueueFamilyIgnored,
                .image               = image,
                .subresourceRange    = {
                    .aspectMask      = vk::ImageAspectFlagBits::eColor,
                    .levelCount      = 1,
                    .baseArrayLayer  = 0,
                    .layerCount      = 1,
                }
            };

        int mipWidth  = static_cast<int>(dimensions.width);
        int mipHeight = static_cast<int>(dimensions.height);

        for (uint32_t i = 1; i < mipLevels; ++i)
        {
            barrier.subresourceRange.baseMipLevel = i - 1;
            barrier.oldLayout                     = vk::ImageLayout::eTransferDstOptimal;
            barrier.newLayout                     = vk::ImageLayout::eTransferSrcOptimal;
            barrier.srcAccessMask                 = vk::AccessFlagBits::eTransferWrite;
            barrier.dstAccessMask                 = vk::AccessFlagBits::eTransferRead;

            commandBuffer->pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                                           vk::PipelineStageFlagBits::eTransfer,
                                           vk::DependencyFlags { 0 },
                                           {},
                                           {},
                                           { barrier });

            // clang-format off
            vk::ImageBlit blit{
                .srcSubresource     = {
                    .aspectMask     = vk::ImageAspectFlagBits::eColor,
                    .mipLevel       = i - 1,
                    .baseArrayLayer = 0,
                    .layerCount     = 1,
                },
                .srcOffsets         = std::array {
                    vk::Offset3D { 0, 0, 0 },
                    vk::Offset3D { mipWidth, mipHeight, 1u },
                },
                .dstSubresource     = {
                    .aspectMask     = vk::ImageAspectFlagBits::eColor,
                    .mipLevel       = i,
                    .baseArrayLayer = 0,
                    .layerCount     = 1
                },
                .dstOffsets         = std::array {
                    vk::Offset3D { 0, 0, 0 },
                    vk::Offset3D { mipWidth  > 1 ? mipWidth  / 2 : 1,
                                   mipHeight > 1 ? mipHeight / 2 : 1,
                                   1 },
                },
            };
            // clang-format on

            commandBuffer->blitImage(image,
                                     vk::ImageLayout::eTransferSrcOptimal,
                                     image,
                                     vk::ImageLayout::eTransferDstOptimal,
                                     { blit },
                                     vk::Filter::eLinear);

            barrier.oldLayout     = vk::ImageLayout::eTransferSrcOptimal;
            barrier.newLayout     = vk::ImageLayout::eShaderReadOnlyOptimal;
            barrier.srcAccessMask = vk::AccessFlagBits::eTransferRead;
            barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

            commandBuffer->pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                                           vk::PipelineStageFlagBits::eFragmentShader,
                                           vk::DependencyFlags { 0 },
                                           {},
                                           {},
                                           { barrier });

            mipWidth /= (mipWidth > 1) ? 2 : 1;
            mipHeight /= (mipHeight > 1) ? 2 : 1;
        }

        barrier.subresourceRange.baseMipLevel = mipLevels - 1;
        barrier.oldLayout                     = vk::ImageLayout::eTransferDstOptimal;
        barrier.newLayout                     = vk::ImageLayout::eShaderReadOnlyOptimal;
        barrier.srcAccessMask                 = vk::AccessFlagBits::eTransferWrite;
        barrier.dstAccessMask                 = vk::AccessFlagBits::eShaderRead;

        commandBuffer->pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                                       vk::PipelineStageFlagBits::eFragmentShader,
                                       vk::DependencyFlags { 0 },
                                       {},
                                       {},
                                       { barrier });
    }
}  // namespace renderer::backend

namespace
{
    using namespace renderer::backend;

    [[maybe_unused]] void transitionImageLayout(ScopedCommandBuffer& commandBuffer,
                                                vk::Image image,
                                                vk::Format format,
                                                vk::ImageLayout oldLayout,
                                                vk::ImageLayout newLayout,
                                                uint32_t mipLevels)
    {
        vk::PipelineStageFlags sourceStage {};
        vk::PipelineStageFlags destinationStage {};

        vk::ImageMemoryBarrier barrier {
            .oldLayout           = oldLayout,
            .newLayout           = newLayout,
            .srcQueueFamilyIndex = vk::QueueFamilyIgnored,
            .dstQueueFamilyIndex = vk::QueueFamilyIgnored,
            .image               = image,
            .subresourceRange    = {
                .aspectMask      = vk::ImageAspectFlagBits::eColor,
                .baseMipLevel    = 0,
                .levelCount      = mipLevels,
                .baseArrayLayer  = 0,
                .layerCount      = 1,
            },
        };

        if (oldLayout == vk::ImageLayout::eUndefined && newLayout == vk::ImageLayout::eTransferDstOptimal)
        {
            barrier.srcAccessMask = vk::AccessFlags { 0 };
            barrier.dstAccessMask = vk::AccessFlagBits::eTransferWrite;

            sourceStage      = vk::PipelineStageFlagBits::eTopOfPipe;
            destinationStage = vk::PipelineStageFlagBits::eTransfer;
        }
        else if (oldLayout == vk::ImageLayout::eTransferDstOptimal &&
                 newLayout == vk::ImageLayout::eShaderReadOnlyOptimal)
        {
            barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
            barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

            sourceStage      = vk::PipelineStageFlagBits::eTransfer;
            destinationStage = vk::PipelineStageFlagBits::eFragmentShader;
        }
        else
        {
            MC_ASSERT_MSG(false, "Unsupported layout transition");
        }

        commandBuffer->pipelineBarrier(
            sourceStage, destinationStage, vk::DependencyFlags { 0 }, {}, {}, { barrier });
    }
}  // namespace
