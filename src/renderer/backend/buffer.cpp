#include "mc/renderer/backend/allocator.hpp"
#include <mc/renderer/backend/buffer.hpp>
#include <mc/renderer/backend/command.hpp>
#include <mc/renderer/backend/vk_checker.hpp>
#include <mc/utils.hpp>
#include <vulkan/vulkan_core.h>

namespace renderer::backend
{
    GPUBuffer::GPUBuffer(Allocator& allocator,
                         size_t allocSize,
                         vk::BufferUsageFlags bufferUsage,
                         VmaMemoryUsage memoryUsage,
                         VmaAllocationCreateFlags allocFlags)
        : m_allocator { &allocator }
    {
        vk::BufferCreateInfo bufferInfo = {
            .size  = allocSize,
            .usage = bufferUsage,
        };

        VmaAllocationCreateInfo vmaAllocInfo = {
            .flags = allocFlags,
            .usage = memoryUsage,
        };

        MC_ASSERT(vmaCreateBuffer(*m_allocator,
                                  &static_cast<VkBufferCreateInfo&>(bufferInfo),
                                  &vmaAllocInfo,
                                  &m_buffer,
                                  &m_allocation,
                                  &m_allocInfo) == VK_SUCCESS);
    }

    GPUBuffer::GPUBuffer(Device& device,
                         Allocator& allocator,
                         std::string_view name,
                         size_t allocSize,
                         vk::BufferUsageFlags bufferUsage,
                         VmaMemoryUsage memoryUsage,
                         VmaAllocationCreateFlags allocFlags)
        : GPUBuffer(allocator, allocSize, bufferUsage, memoryUsage, allocFlags)
    {
        setName(device, name);
    };

    GPUBuffer::~GPUBuffer()
    {
        if (m_buffer == nullptr)
        {
            return;
        }

        vmaDestroyBuffer(*m_allocator, m_buffer, m_allocation);

        m_buffer = nullptr;
    }
}  // namespace renderer::backend
