#include "mc/renderer/backend/allocator.hpp"
#include <mc/renderer/backend/buffer.hpp>
#include <mc/renderer/backend/command.hpp>
#include <mc/renderer/backend/vk_checker.hpp>
#include <mc/utils.hpp>
#include <vulkan/vulkan_core.h>

namespace renderer::backend
{
    GPUBuffer::GPUBuffer(ResourceHandle const& handle,
                         std::string const& name,
                         Device& device,
                         Allocator& allocator,
                         size_t allocSize,
                         vk::BufferUsageFlags bufferUsage,
                         VmaMemoryUsage memoryUsage,
                         VmaAllocationCreateFlags allocFlags)
        : ResourceBase { handle }, device { &device }, allocator { &allocator }
    {
        vk::BufferCreateInfo bufferInfo = {
            .size  = allocSize,
            .usage = bufferUsage,
        };

        VmaAllocationCreateInfo vmaAllocInfo = {
            .flags = allocFlags,
            .usage = memoryUsage,
        };

        MC_ASSERT(vmaCreateBuffer(allocator.get(),
                                  &static_cast<VkBufferCreateInfo&>(bufferInfo),
                                  &vmaAllocInfo,
                                  &vulkanHandle,
                                  &allocation,
                                  &allocInfo) == VK_SUCCESS);
        setName(name);
    }

    GPUBuffer::~GPUBuffer()
    {
        if (vulkanHandle == nullptr)
        {
            return;
        }

        vmaDestroyBuffer(*allocator, vulkanHandle, allocation);

        vulkanHandle = nullptr;
    }
}  // namespace renderer::backend
