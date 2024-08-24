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

        GPUBuffer(GPUBuffer const&)                    = delete;
        auto operator=(GPUBuffer const&) -> GPUBuffer& = delete;

        auto operator=(GPUBuffer&& other) noexcept -> GPUBuffer&
        {
            if (*this == other || !other.m_allocation)
            {
                return *this;
            }

            m_buffer     = other.m_buffer;
            m_allocator  = other.m_allocator;
            m_allocation = other.m_allocation;
            m_allocInfo  = other.m_allocInfo;

            other.m_buffer     = VK_NULL_HANDLE;
            other.m_allocation = nullptr;
            other.m_allocInfo  = {};

            return *this;
        }

        GPUBuffer(GPUBuffer&& other) noexcept
            : m_allocator { other.m_allocator },
              m_buffer { other.m_buffer },
              m_allocation { other.m_allocation },
              m_allocInfo { other.m_allocInfo }
        {
            other.m_buffer     = VK_NULL_HANDLE;
            other.m_allocation = nullptr;
            other.m_allocInfo  = {};
        };

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
