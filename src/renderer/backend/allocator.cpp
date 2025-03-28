#include <mc/renderer/backend/allocator.hpp>

namespace renderer::backend
{
    Allocator::Allocator(Instance const& instance, Device const& device)
    {
        VmaAllocatorCreateInfo allocatorInfo = {
            .flags          = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT,
            .physicalDevice = *device.getPhysical(),
            .device         = *device.get(),
            .instance       = static_cast<vk::Instance>(instance),
        };

        vmaCreateAllocator(&allocatorInfo, &m_allocator);
    }

    Allocator::~Allocator()
    {
        if (m_allocator)
        {
            vmaDestroyAllocator(m_allocator);
        }
    }
}  // namespace renderer::backend
