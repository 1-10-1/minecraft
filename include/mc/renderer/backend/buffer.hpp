#pragma once

#include "allocator.hpp"
#include <vulkan/vulkan_core.h>

#if DEBUG
#    include "vk_checker.hpp"
#endif

#include <vk_mem_alloc.h>
#include <vulkan/vulkan_raii.hpp>

namespace renderer::backend
{
    // FIXME(aether) This name is misleading, GPUBuffer can be a cpu-only staging buffer
    class GPUBuffer
    {
    public:
        GPUBuffer() = default;

        GPUBuffer(Allocator& allocator,
                  size_t allocSize,
                  vk::BufferUsageFlags bufferUsage,
                  VmaMemoryUsage memoryUsage,
                  VmaAllocationCreateFlags allocFlags = 0);

        // Use this constructor to set a name
        GPUBuffer(Device& device,
                  Allocator& allocator,
                  std::string_view name,
                  size_t allocSize,
                  vk::BufferUsageFlags bufferUsage,
                  VmaMemoryUsage memoryUsage,
                  VmaAllocationCreateFlags allocFlags = 0);

        ~GPUBuffer();

        friend void swap(GPUBuffer& first, GPUBuffer& second) noexcept
        {
            using std::swap;

            swap(first.m_buffer, second.m_buffer);
            swap(first.m_allocator, second.m_allocator);
            swap(first.m_allocInfo, second.m_allocInfo);
            swap(first.m_allocation, second.m_allocation);
        }

        GPUBuffer(GPUBuffer&& other) noexcept : GPUBuffer() { swap(*this, other); };

        GPUBuffer& operator=(GPUBuffer other) noexcept
        {
            swap(*this, other);

            return *this;
        }

        [[nodiscard]] operator bool() const { return m_buffer; }

        [[nodiscard]] bool operator==(std::nullptr_t) const { return !m_buffer; }

        [[nodiscard]] operator vk::Buffer() const { return m_buffer; }

        [[nodiscard]] auto operator->() const -> vk::Buffer { return m_buffer; }

        [[nodiscard]] auto get() const -> vk::Buffer { return m_buffer; }

        [[nodiscard]] auto getName() const -> std::string_view
        {
#if DEBUG
            VmaAllocationInfo allocInfo;
            vmaGetAllocationInfo(*m_allocator, m_allocation, &allocInfo);

            return allocInfo.pName;
#else
            return "";
#endif
        }

        void setName(Device const& device, std::string_view name)
        {
#if DEBUG
            vmaSetAllocationName(*m_allocator, m_allocation, name.data());

            auto func = reinterpret_cast<PFN_vkSetDebugUtilsObjectNameEXT>(
                device.getInstance().getProcAddr("vkSetDebugUtilsObjectNameEXT"));

            VkDebugUtilsObjectNameInfoEXT info {
                .sType        = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT,
                .objectType   = VK_OBJECT_TYPE_BUFFER,
                .objectHandle = reinterpret_cast<uint64_t>(m_buffer),
                .pObjectName  = name.data(),
            };

            func(*device.get(), &info) >> ResultChecker();
#endif
        }

        [[nodiscard]] auto getMappedData() const -> void* { return m_allocInfo.pMappedData; }

        [[nodiscard]] auto getSize() const -> size_t { return m_allocInfo.size; }

    private:
        Allocator* m_allocator { nullptr };

        VkBuffer m_buffer { VK_NULL_HANDLE };
        VmaAllocation m_allocation { nullptr };
        VmaAllocationInfo m_allocInfo {};
    };
}  // namespace renderer::backend
