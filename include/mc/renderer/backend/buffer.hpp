#pragma once

#include "command.hpp"
#include "device.hpp"
#include "ubo.hpp"

#include <vulkan/vulkan.h>
#include <vulkan/vulkan_core.h>

namespace renderer::backend
{
    class StagedBuffer
    {
    public:
        StagedBuffer(Device& device,
                     CommandManager& commandManager,
                     VkBufferUsageFlags usageFlags,
                     void const* data,
                     size_t sizeInBytes);

        StagedBuffer(StagedBuffer const&)                    = delete;
        StagedBuffer(StagedBuffer&&)                         = delete;
        auto operator=(StagedBuffer const&) -> StagedBuffer& = delete;
        auto operator=(StagedBuffer&&) -> StagedBuffer&      = delete;

        ~StagedBuffer()
        {
            vkDestroyBuffer(m_device, m_bufferHandle, nullptr);
            vkFreeMemory(m_device, m_memoryHandle, nullptr);
        }

        // NOLINTNEXTLINE(google-explicit-constructor)
        [[nodiscard]] operator VkBuffer() const { return m_bufferHandle; }

    private:
        Device& m_device;

        VkBuffer m_bufferHandle { VK_NULL_HANDLE };
        VkDeviceMemory m_memoryHandle { VK_NULL_HANDLE };
    };

    class UniformBuffer
    {
    public:
        UniformBuffer(Device& device, CommandManager& commandController);

        UniformBuffer(UniformBuffer const&)                    = delete;
        UniformBuffer(UniformBuffer&&)                         = delete;
        auto operator=(UniformBuffer const&) -> UniformBuffer& = delete;
        auto operator=(UniformBuffer&&) -> UniformBuffer&      = delete;

        ~UniformBuffer()
        {
            vkDestroyBuffer(m_device, m_bufferHandle, nullptr);
            vkFreeMemory(m_device, m_memoryHandle, nullptr);
        }

        void update(UniformBufferObject const& ubo);

    private:
        Device& m_device;

        void* m_bufferMapping { nullptr };

        VkBuffer m_bufferHandle { VK_NULL_HANDLE };
        VkDeviceMemory m_memoryHandle { VK_NULL_HANDLE };
    };

    class TextureBuffer
    {
    public:
        TextureBuffer(Device& device, CommandManager& commandController);

        TextureBuffer(TextureBuffer const&)                    = delete;
        TextureBuffer(TextureBuffer&&)                         = delete;
        auto operator=(TextureBuffer const&) -> TextureBuffer& = delete;
        auto operator=(TextureBuffer&&) -> TextureBuffer&      = delete;

        ~TextureBuffer()
        {
            vkDestroyBuffer(m_device, m_bufferHandle, nullptr);
            vkFreeMemory(m_device, m_memoryHandle, nullptr);
        }

        void init(void const* data, uint64_t sizeInBytes);

    private:
        Device& m_device;

        VkBuffer m_bufferHandle { VK_NULL_HANDLE };
        VkDeviceMemory m_memoryHandle { VK_NULL_HANDLE };
    };
}  // namespace renderer::backend