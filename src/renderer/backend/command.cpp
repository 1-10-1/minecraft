#include "mc/asserts.hpp"
#include <algorithm>
#include <mc/renderer/backend/command.hpp>
#include <mc/renderer/backend/constants.hpp>
#include <mc/renderer/backend/info_structs.hpp>
#include <mc/renderer/backend/vk_checker.hpp>
#include <mc/utils.hpp>

#include <ranges>
#include <vulkan/vulkan_core.h>

namespace rn = std::ranges;
namespace vi = std::ranges::views;

namespace renderer::backend
{
    ScopedCommandBuffer::ScopedCommandBuffer(Device& device,
                                             vk::raii::CommandPool const& commandPool,
                                             vk::raii::Queue const& queue,
                                             bool oneTimeUse)
        : m_device { &device }, m_oneTime { oneTimeUse }, m_queue { queue }, m_pool { commandPool }
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

    void ScopedCommandBuffer::flush()
    {
        m_handle.end();

        std::array cmdSubmits { vk::CommandBufferSubmitInfo().setCommandBuffer(m_handle) };
        std::array submits { vk::SubmitInfo2().setCommandBufferInfos(cmdSubmits) };

        vk::raii::Fence fence = m_device->get().createFence(vk::FenceCreateInfo {}).value();

        MC_ASSERT(m_queue.submit2(submits, fence) == vk::Result::eSuccess);

        MC_ASSERT(m_device->get().waitForFences({ fence }, true, std::numeric_limits<uint64_t>::max()) !=
                  vk::Result::eTimeout);

        if (m_oneTime)
        {
            m_handle = std::move(m_device->get()
                                     .allocateCommandBuffers(vk::CommandBufferAllocateInfo()
                                                                 .setCommandPool(m_pool)
                                                                 .setLevel(vk::CommandBufferLevel::ePrimary)
                                                                 .setCommandBufferCount(1))
                                     .value()[0]);

            m_handle.begin(
                vk::CommandBufferBeginInfo().setFlags(vk::CommandBufferUsageFlagBits::eOneTimeSubmit));
        }
        else
        {
            m_handle.begin(vk::CommandBufferBeginInfo());
        }
    }

    CommandManager::CommandManager(Device const& device, uint32_t numThreads)
        : numPoolsPerFrame { numThreads }
    {
        m_mainCommandPool =
            device
                ->createCommandPool(vk::CommandPoolCreateInfo()
                                        .setFlags(vk::CommandPoolCreateFlagBits::eResetCommandBuffer)
                                        .setQueueFamilyIndex(device.getQueueFamilyIndices().mainFamily))
                .value();

        m_transferCommandPool =
            device
                ->createCommandPool(vk::CommandPoolCreateInfo()
                                        .setFlags(vk::CommandPoolCreateFlagBits::eResetCommandBuffer |
                                                  vk::CommandPoolCreateFlagBits::eTransient)
                                        .setQueueFamilyIndex(device.getQueueFamilyIndices().transferFamily))
                .value();

        uint32_t const totalPools = numPoolsPerFrame * kNumFramesInFlight;

        for (uint32_t _ : vi::iota(0u, totalPools))
        {
            commandPools.push_back(device->createCommandPool(
                                       vk::CommandPoolCreateInfo()
                                           .setQueueFamilyIndex(device.getQueueFamilyIndices().mainFamily)
                                           .setFlags(vk::CommandPoolCreateFlagBits::eResetCommandBuffer)) >>
                                   ResultChecker());
        }

        usedBuffers.resize(totalPools);
        usedSecondaryBuffers.resize(totalPools);

        // Create command buffers: pools * buffers per pool
        uint32_t const totalPrimaryBuffers   = totalPools * numCommandBuffersPerThread;
        uint32_t const totalSecondaryBuffers = totalPools * kNumSecondaryBuffers;

        primaryBuffers.reserve(totalPrimaryBuffers);
        secondaryBuffers.reserve(totalSecondaryBuffers);

        for (uint32_t i : vi::iota(0u, totalPrimaryBuffers))
        {
            uint32_t const frame_index  = i / (numCommandBuffersPerThread * numPoolsPerFrame);
            uint32_t const thread_index = (i / numCommandBuffersPerThread) % numPoolsPerFrame;
            uint32_t const pool_index   = poolFromIndices(frame_index, thread_index);

            i++;

            primaryBuffers.push_back(
                std::move((device->allocateCommandBuffers(vk::CommandBufferAllocateInfo()
                                                              .setCommandPool(commandPools[pool_index])
                                                              .setLevel(vk::CommandBufferLevel::ePrimary)
                                                              .setCommandBufferCount(1)) >>
                           ResultChecker())[0]));
        };

        for (uint32_t i = 0; i < totalPrimaryBuffers; i++)

            for (uint32_t poolIndex : vi::iota(0u, totalPools))
            {
                rn::move(device->allocateCommandBuffers(vk::CommandBufferAllocateInfo()
                                                            .setCommandPool(commandPools[poolIndex])
                                                            .setLevel(vk::CommandBufferLevel::eSecondary)
                                                            .setCommandBufferCount(kNumSecondaryBuffers)) >>
                             ResultChecker(),
                         std::back_inserter(secondaryBuffers));
            }
    }

    void CommandManager::resetPools(uint32_t frameIndex)
    {
        for (uint32_t i : vi::iota(0u, numPoolsPerFrame))
        {
            uint32_t const poolIndex = poolFromIndices(frameIndex, i);

            commandPools[poolIndex].reset();
            usedBuffers[poolIndex]          = 0;
            usedSecondaryBuffers[poolIndex] = 0;
        }
    }

    vk::CommandBuffer CommandManager::getCommandBuffer(uint32_t frame, uint32_t threadIndex, bool begin)
    {
        uint32_t const poolIndex   = poolFromIndices(frame, threadIndex);
        uint32_t currentUsedBuffer = usedBuffers[poolIndex];

        // TODO: how to handle fire-and-forget command buffers ?
        //used_buffers[ pool_index ] = current_used_buffer + 1;

        MC_ASSERT(currentUsedBuffer < numCommandBuffersPerThread);

        auto& cb = primaryBuffers[(poolIndex * numCommandBuffersPerThread) + currentUsedBuffer];

        if (begin)
        {
            cb.reset();

            cb.begin(vk::CommandBufferBeginInfo().setFlags(vk::CommandBufferUsageFlagBits::eOneTimeSubmit));
        }

        return cb;
    }

    vk::CommandBuffer CommandManager::getSecondaryCommandBuffer(uint32_t frame, uint32_t threadIndex)
    {
        uint32_t const poolIndex        = poolFromIndices(frame, threadIndex);
        uint32_t currentUsedBuffer      = usedSecondaryBuffers[poolIndex];
        usedSecondaryBuffers[poolIndex] = currentUsedBuffer + 1;

        MC_ASSERT(currentUsedBuffer < kNumSecondaryBuffers);

        return secondaryBuffers[(poolIndex * kNumSecondaryBuffers) + currentUsedBuffer];
    }
}  // namespace renderer::backend
