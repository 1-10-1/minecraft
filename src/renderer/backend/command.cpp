#include "mc/asserts.hpp"
#include <mc/renderer/backend/command.hpp>
#include <mc/renderer/backend/constants.hpp>
#include <mc/renderer/backend/info_structs.hpp>
#include <mc/renderer/backend/vk_checker.hpp>
#include <mc/utils.hpp>

#include <vulkan/vulkan_core.h>

namespace renderer::backend
{
    ScopedCommandBuffer::ScopedCommandBuffer(Device& device,
                                             vk::raii::CommandPool const& commandPool,
                                             vk::raii::Queue const& queue,
                                             bool oneTimeUse)
        : m_device { &device }, m_queue { queue }
    {
        m_handle = std::move(m_device->get()
                                 .allocateCommandBuffers(vk::CommandBufferAllocateInfo()
                                                             .setCommandPool(commandPool)
                                                             .setLevel(vk::CommandBufferLevel::ePrimary)
                                                             .setCommandBufferCount(1))
                                 .value()[0]);

        m_handle.begin(
            vk::CommandBufferBeginInfo().setFlags(oneTimeUse ? vk::CommandBufferUsageFlagBits::eOneTimeSubmit
                                                             : static_cast<vk::CommandBufferUsageFlags>(0)));
    }

    ScopedCommandBuffer::~ScopedCommandBuffer()
    {
        if (!*m_handle)
        {
            return;
        }

        m_handle.end();

        std::array cmdSubmits { vk::CommandBufferSubmitInfo().setCommandBuffer(m_handle) };
        std::array submits { vk::SubmitInfo2().setCommandBufferInfos(cmdSubmits) };

        vk::raii::Fence fence = m_device->get().createFence(vk::FenceCreateInfo {}).value();

        MC_ASSERT(m_queue.submit2(submits, fence) == vk::Result::eSuccess);

        MC_ASSERT(m_device->get().waitForFences({ fence }, true, std::numeric_limits<uint64_t>::max()) !=
                  vk::Result::eTimeout);
    }

    CommandManager::CommandManager(Device const& device)
    {
        m_graphicsCommandPool =
            device
                ->createCommandPool(vk::CommandPoolCreateInfo()
                                        .setFlags(vk::CommandPoolCreateFlagBits::eResetCommandBuffer)
                                        .setQueueFamilyIndex(device.getQueueFamilyIndices().graphicsFamily))
                .value();

        m_transferCommandPool =
            device
                ->createCommandPool(vk::CommandPoolCreateInfo()
                                        .setFlags(vk::CommandPoolCreateFlagBits::eResetCommandBuffer |
                                                  vk::CommandPoolCreateFlagBits::eTransient)
                                        .setQueueFamilyIndex(device.getQueueFamilyIndices().transferFamily))
                .value();

        m_graphicsCommandBuffers =
            device
                ->allocateCommandBuffers(vk::CommandBufferAllocateInfo()
                                             .setCommandPool(m_graphicsCommandPool)
                                             .setLevel(vk::CommandBufferLevel::ePrimary)
                                             .setCommandBufferCount(kNumFramesInFlight))
                .value();
    }
}  // namespace renderer::backend
