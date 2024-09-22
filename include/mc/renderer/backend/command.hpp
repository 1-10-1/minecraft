#pragma once

#include "device.hpp"

#include <utility>
#include <vulkan/vulkan_raii.hpp>
#include <vulkan/vulkan_shared.hpp>

namespace renderer::backend
{
    class ScopedCommandBuffer
    {
    public:
        ScopedCommandBuffer() = default;

        ScopedCommandBuffer(Device& device,
                            vk::raii::CommandPool const& commandPool,
                            vk::raii::Queue const& queue,
                            bool oneTimeUse = false);
        ~ScopedCommandBuffer();

        friend void swap(ScopedCommandBuffer& first, ScopedCommandBuffer& second) noexcept
        {
            using std::swap;

            swap(first.m_device, second.m_device);
            swap(first.m_handle, second.m_handle);
            swap(first.m_pool, second.m_pool);
            swap(first.m_oneTime, second.m_oneTime);
            swap(first.m_queue, second.m_queue);
        }

        ScopedCommandBuffer(ScopedCommandBuffer&& other) noexcept : ScopedCommandBuffer()
        {
            swap(*this, other);
        };

        ScopedCommandBuffer& operator=(ScopedCommandBuffer other) noexcept
        {
            swap(*this, other);

            return *this;
        }

        [[nodiscard]] operator vk::CommandBuffer() const { return m_handle; }

        [[nodiscard]] auto operator->() const -> vk::raii::CommandBuffer const* { return &m_handle; }

        void flush();

    private:
        Device const* m_device { nullptr };

        bool m_oneTime { false };

        vk::Queue m_queue { nullptr };

        vk::CommandPool m_pool { nullptr };
        vk::raii::CommandBuffer m_handle { nullptr };
    };

    class CommandManager
    {
    public:
        CommandManager()  = default;
        ~CommandManager() = default;

        explicit CommandManager(Device const& device, uint32_t numThreads);

        friend void swap(CommandManager& first, CommandManager& second) noexcept
        {
            using std::swap;

            swap(first.m_mainCommandPool, second.m_mainCommandPool);
            swap(first.m_transferCommandPool, second.m_transferCommandPool);
            swap(first.commandPools, second.commandPools);
            swap(first.primaryBuffers, second.primaryBuffers);
            swap(first.secondaryBuffers, second.secondaryBuffers);
            swap(first.usedBuffers, second.usedBuffers);
            swap(first.usedSecondaryBuffers, second.usedSecondaryBuffers);
            swap(first.numPoolsPerFrame, second.numPoolsPerFrame);
            swap(first.numCommandBuffersPerThread, second.numCommandBuffersPerThread);
        }

        CommandManager(CommandManager&& other) noexcept : CommandManager() { swap(*this, other); };

        CommandManager& operator=(CommandManager other) noexcept
        {
            swap(*this, other);

            return *this;
        }

        [[nodiscard]] auto getMainCmdPool() const -> vk::raii::CommandPool const&
        {
            return m_mainCommandPool;
        }

        [[nodiscard]] auto getTransferCmdPool() const -> vk::raii::CommandPool const&
        {
            return m_transferCommandPool;
        }

        void resetPools(uint32_t frameIndex);

        vk::CommandBuffer getCommandBuffer(uint32_t frame, uint32_t threadIndex, bool begin);
        vk::CommandBuffer getSecondaryCommandBuffer(uint32_t frame, uint32_t threadIndex);

    private:
        vk::raii::CommandPool m_mainCommandPool { nullptr };
        vk::raii::CommandPool m_transferCommandPool { nullptr };

        uint32_t poolFromIndices(uint32_t frameIndex, uint32_t threadIndex)
        {
            return (frameIndex * numPoolsPerFrame) + threadIndex;
        }

        std::vector<vk::raii::CommandPool> commandPools;
        std::vector<vk::raii::CommandBuffer> primaryBuffers;
        std::vector<vk::raii::CommandBuffer> secondaryBuffers;
        std::vector<uint8_t> usedBuffers;  // Track how many buffers were used per thread per frame.
        std::vector<uint8_t> usedSecondaryBuffers;

        uint32_t numPoolsPerFrame           = 0;
        uint32_t numCommandBuffersPerThread = 3;
    };
}  // namespace renderer::backend
