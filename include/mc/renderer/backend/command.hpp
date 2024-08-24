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

        ScopedCommandBuffer(ScopedCommandBuffer const&)                    = delete;
        auto operator=(ScopedCommandBuffer const&) -> ScopedCommandBuffer& = delete;

        auto operator=(ScopedCommandBuffer&& other) noexcept -> ScopedCommandBuffer&
        {
            if (this == &other)
            {
                return *this;
            }

            m_device = std::exchange(other.m_device, nullptr);
            m_queue  = std::exchange(other.m_queue, nullptr);
            m_handle = std::exchange(other.m_handle, nullptr);

            return *this;
        };

        ScopedCommandBuffer(ScopedCommandBuffer&& other) noexcept
            : m_device { std::exchange(other.m_device, nullptr) },
              m_queue { std::exchange(other.m_queue, nullptr) },
              m_handle { std::exchange(other.m_handle, nullptr) } {};

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

        explicit CommandManager(Device const& device);

        CommandManager(CommandManager const&)                    = delete;
        auto operator=(CommandManager const&) -> CommandManager& = delete;

        auto operator=(CommandManager&& other) noexcept -> CommandManager&
        {
            if (this == &other)
            {
                return *this;
            }

            m_mainCommandPool     = std::exchange(other.m_mainCommandPool, nullptr);
            m_transferCommandPool = std::exchange(other.m_transferCommandPool, nullptr);
            m_mainCommandBuffers  = std::exchange(other.m_mainCommandBuffers, {});

            return *this;
        };

        CommandManager(CommandManager&& other) noexcept
            : m_mainCommandPool { std::exchange(other.m_mainCommandPool, nullptr) },
              m_transferCommandPool { std::exchange(other.m_transferCommandPool, nullptr) },
              m_mainCommandBuffers { std::exchange(other.m_mainCommandBuffers, {}) }
        {
        }

        [[nodiscard]] auto getMainCmdBuffer(size_t index) const -> vk::raii::CommandBuffer const&
        {
            return m_mainCommandBuffers[index];
        }

        [[nodiscard]] auto getMainCmdPool() const -> vk::raii::CommandPool const&
        {
            return m_mainCommandPool;
        }

        [[nodiscard]] auto getTransferCmdPool() const -> vk::raii::CommandPool const&
        {
            return m_transferCommandPool;
        }

    private:
        vk::raii::CommandPool m_mainCommandPool { nullptr };
        vk::raii::CommandPool m_transferCommandPool { nullptr };

        std::vector<vk::raii::CommandBuffer> m_mainCommandBuffers {};
    };
}  // namespace renderer::backend
