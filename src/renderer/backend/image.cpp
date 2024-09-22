#include <mc/asserts.hpp>
#include <mc/exceptions.hpp>
#include <mc/renderer/backend/allocator.hpp>
#include <mc/renderer/backend/buffer.hpp>
#include <mc/renderer/backend/command.hpp>
#include <mc/renderer/backend/image.hpp>
#include <mc/renderer/backend/vk_checker.hpp>

#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_core.h>
#include <vulkan/vulkan_structs.hpp>

namespace renderer::backend
{
    Image::Image(ResourceHandle const& handle,
                 std::string const& name [[maybe_unused]],
                 Device const& device,
                 Allocator const& allocator,
                 vk::Extent2D dimensions,
                 vk::Format format,
                 vk::SampleCountFlagBits sampleCount,
                 vk::ImageUsageFlags usageFlags,
                 vk::ImageAspectFlags aspectFlags,
                 uint32_t mipLevels)
        : ResourceBase { handle },
          device { &device },
          allocator { &allocator },
          format { format },
          sampleCount { sampleCount },
          usageFlags { usageFlags },
          aspectFlags { aspectFlags },
          mipLevels { mipLevels },
          dimensions { dimensions }
    {
        create();

#if DEBUG
        setName(name);
#endif
    }

    Image::~Image()
    {
        destroy();
    };

    void Image::create()
    {
        createImage(format,
                    vk::ImageTiling::eOptimal,
                    usageFlags,
                    vk::MemoryPropertyFlagBits::eDeviceLocal,
                    mipLevels,
                    sampleCount);

        // If the image is solely being used for transfer, dont make a view
        if ((usageFlags & (vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst)) <
            usageFlags)
        {
            createImageView(format, aspectFlags, 1);
        }
    }

    void Image::destroy()
    {
        if (!imageHandle)
        {
            return;
        }

        imageView.clear();
        vmaDestroyImage(*allocator, imageHandle, allocation);

        imageHandle = nullptr;
    }

    void Image::setName(std::string const& newName)
    {
#if DEBUG
        vmaSetAllocationName(*allocator, allocation, newName.data());

        VmaAllocationInfo allocInfo;
        vmaGetAllocationInfo(*allocator, allocation, &allocInfo);
        name = allocInfo.pName;

        auto func = reinterpret_cast<PFN_vkSetDebugUtilsObjectNameEXT>(
            device->getInstance().getProcAddr("vkSetDebugUtilsObjectNameEXT"));

        VkDebugUtilsObjectNameInfoEXT info {
            .sType        = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT,
            .objectType   = VK_OBJECT_TYPE_IMAGE,
            .objectHandle = reinterpret_cast<uint64_t>(imageHandle),
            .pObjectName  = newName.data(),
        };

        func(*device->get(), &info) >> ResultChecker();
#endif
    }

    void ResourceAccessor<Image>::copyTo(vk::CommandBuffer cmdBuf,
                                         vk::Image dst,
                                         vk::Extent2D dstSize,
                                         vk::Extent2D offset)
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

        blitInfo.srcImage       = get().imageHandle;
        blitInfo.srcImageLayout = vk::ImageLayout::eTransferSrcOptimal;

        blitInfo.filter      = vk::Filter::eLinear;
        blitInfo.regionCount = 1;
        blitInfo.pRegions    = &blitRegion;

        cmdBuf.blitImage2(blitInfo);
    };

    auto ResourceAccessor<Image>::getName() const -> std::string_view
    {
#if DEBUG
        return get().name;
#else
        return "";
#endif
    }

    void Image::transition(vk::CommandBuffer cmdBuf,
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

    void Image::createImage(vk::Format format,
                            vk::ImageTiling tiling,
                            vk::ImageUsageFlags usage,
                            vk::MemoryPropertyFlags properties,
                            uint32_t mipLevels,
                            vk::SampleCountFlagBits numSamples)
    {
        vk::ImageCreateInfo imageInfo {
            .imageType     = vk::ImageType::e2D,
            .format        = format,
            .extent        = { dimensions.width, dimensions.height, 1 },
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

        vmaCreateImage(*allocator,
                       &static_cast<VkImageCreateInfo&>(imageInfo),
                       &imageAllocInfo,
                       &imageHandle,
                       &allocation,
                       nullptr);
    }

    void Image::createImageView(vk::Format format, vk::ImageAspectFlags aspectFlags, uint32_t mipLevels)
    {
        vk::ImageViewCreateInfo viewInfo {
            .image              = imageHandle,
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

        imageView = device->get().createImageView(viewInfo) >> ResultChecker();
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
