#include <mc/renderer/backend/surface.hpp>

#include <algorithm>

#include <mc/exceptions.hpp>
#include <vulkan/vulkan_core.h>

namespace renderer::backend
{
    void Surface::refresh(vk::PhysicalDevice device)
    {
        m_details.capabilities = device.getSurfaceCapabilitiesKHR(m_surface).value;
        m_details.formats      = device.getSurfaceFormatsKHR(m_surface).value;
        m_details.presentModes = device.getSurfacePresentModesKHR(m_surface).value;

        MC_ASSERT(m_details.formats.size() > 0 && m_details.presentModes.size() > 0);

        // Choose extent
        {
            if (m_details.capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max())
            {
                m_details.extent = m_details.capabilities.currentExtent;
            }
            else
            {
                glm::uvec2 dimensions = m_window->getFramebufferDimensions();

                m_details.extent = vk::Extent2D {
                    .width = std::clamp(dimensions.x,
                                        m_details.capabilities.minImageExtent.width,
                                        m_details.capabilities.maxImageExtent.width),

                    .height = std::clamp(dimensions.y,
                                         m_details.capabilities.minImageExtent.height,
                                         m_details.capabilities.maxImageExtent.height),
                };
            }
        }

        // Choose format
        {
            m_details.surfaceFormat = m_details.formats[0];

            for (auto const& availableFormat : m_details.formats)
            {
                if (availableFormat.format == vk::Format::eB8G8R8A8Unorm &&
                    availableFormat.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear)
                {
                    m_details.surfaceFormat = availableFormat;
                }
            }

            m_details.format = m_details.surfaceFormat.format;
        }

        // Choose present mode
        {
            bool presentModeChosen = false;

            if (!m_vsync)
            {
                for (auto const& availablePresentMode : m_details.presentModes)
                {
                    if (availablePresentMode == vk::PresentModeKHR::eImmediate)
                    {
                        m_details.presentMode = availablePresentMode;
                        presentModeChosen     = true;
                    }
                }
            }

            if (!presentModeChosen)
            {
                m_details.presentMode = vk::PresentModeKHR::eFifo;
            }
        }
    }
}  // namespace renderer::backend
