#include <mc/renderer/backend/swapchain.hpp>
#include <mc/renderer/backend/vk_checker.hpp>

#include <vulkan/vulkan_core.h>

#include <ranges>

namespace renderer::backend
{
    Swapchain::Swapchain(Device const& device, Surface& surface, bool refreshSurface) : m_device { &device }
    {
        if (refreshSurface)
        {
            surface.refresh(m_device->getPhysical());
        }

        auto const& details = surface.getDetails();

        m_imageExtent = details.extent;

        std::uint32_t imageCount = details.capabilities.minImageCount + 1;

        if (details.capabilities.maxImageCount > 0 && imageCount > details.capabilities.maxImageCount)
        {
            imageCount = details.capabilities.maxImageCount;
        }

        QueueFamilyIndices const& queueFamilyIndices = m_device->getQueueFamilyIndices();

        std::array queueFamilyIndicesArray { queueFamilyIndices.mainFamily,
                                             queueFamilyIndices.presentFamily };

        bool sameQueueFamily = queueFamilyIndices.mainFamily == queueFamilyIndices.presentFamily;

        vk::SwapchainCreateInfoKHR createInfo {
            .surface          = surface,
            .minImageCount    = imageCount,
            .imageFormat      = details.surfaceFormat.format,
            .imageColorSpace  = details.surfaceFormat.colorSpace,
            .imageExtent      = details.extent,
            .imageArrayLayers = 1,
            .imageUsage = vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eColorAttachment,
            .imageSharingMode = sameQueueFamily ? vk::SharingMode::eExclusive : vk::SharingMode::eConcurrent,
            .queueFamilyIndexCount = sameQueueFamily ? 0u : 2u,
            .pQueueFamilyIndices   = sameQueueFamily ? nullptr : queueFamilyIndicesArray.data(),
            .preTransform          = details.capabilities.currentTransform,
            .compositeAlpha        = vk::CompositeAlphaFlagBitsKHR::eOpaque,
            .presentMode           = details.presentMode,
            .clipped               = true,
        };

        m_handle = m_device->get().createSwapchainKHR(createInfo) >> ResultChecker();

        m_images = m_handle.getImages();

        m_imageViews.reserve(imageCount);

        for (size_t i : vi::iota(0u, imageCount))
        {
            m_imageViews.push_back(
                (*m_device)
                    ->createImageView(
                        vk::ImageViewCreateInfo()
                            .setImage(m_images[i])
                            .setViewType(vk::ImageViewType::e2D)
                            .setFormat(details.format)
                            .setComponents(vk::ComponentMapping {
                                .r = vk::ComponentSwizzle::eIdentity,
                                .g = vk::ComponentSwizzle::eIdentity,
                                .b = vk::ComponentSwizzle::eIdentity,
                                .a = vk::ComponentSwizzle::eIdentity,
                            })
                            .setSubresourceRange({ .aspectMask     = vk::ImageAspectFlagBits::eColor,
                                                   .baseMipLevel   = 0,
                                                   .levelCount     = 1,
                                                   .baseArrayLayer = 0,
                                                   .layerCount     = 1 }))
                    .value());
        }
    }
}  // namespace renderer::backend
